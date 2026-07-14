'''
Module to resolve MERRA-2 granules for zagg events and build zagg payloads.

The zagg temporal pipeline (englacial/zagg#213) takes events either as
in-memory tuples (local backend) or URI dicts (Lambda backend); both need to
know which MERRA-2 granules cover each storm. This module queries earthaccess
ONCE per collection DOI across the whole catalog span, caches a
{collection: {date: url}} index as JSON, and maps individual storms (rows of
the zagg_events.export_events catalog) onto their granule URLs and payloads.
Precipitation runs aggregate under the 24-hour augmented masks, so their
granule ranges resolve against t_end_augmented rather than t_end.

Jimmy Butler & co.
July 2026
'''

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# granule-index cache layout version: bumped when the on-disk shape changes
# (v2: each date maps to {"s3": url, "https": url} instead of one url)
_CACHE_VERSION = 2

# MERRA-2 collection DOIs, keyed by the collection names the zagg configs use.
# merra2_flx and merra2_precip share a DOI: same source files, two processing
# branches (vIVT reads raw rates; precip scales/resamples to 3-hourly totals).
MERRA2_DOIS = {
    'merra2_slv': '10.5067/3Z173KIE2TPD',     # T2M, TQV, SLP
    'merra2_flx': '10.5067/Q5GVUVUIVGO7',     # VFLXQV
    'merra2_precip': '10.5067/Q5GVUVUIVGO7',  # PRECCU, PRECLS, PRECSN
    'merra2_v850': '10.5067/VJAFPLI1CSIV',    # V850
    'merra2_omega': '10.5067/QBZ6MG944HW0',   # OMEGA (on levels)
}


def _granule_date(granule):
    '''Extract the YYYY-MM-DD date of an earthaccess granule.'''
    umm = granule.get('umm', granule)
    try:
        return umm['TemporalExtent']['RangeDateTime']['BeginningDateTime'][:10]
    except (KeyError, TypeError):
        pass
    # fallback: MERRA-2 filenames carry YYYYMMDD as a dot segment,
    # e.g. MERRA2_400.tavg1_2d_slv_Nx.20200115.nc4
    for link in granule.data_links():
        for part in link.split('.'):
            if len(part) == 8 and part.isdigit():
                return f'{part[:4]}-{part[4:6]}-{part[6:]}'
    raise ValueError(f'could not extract a date from granule: {granule}')


def _granule_urls(granule):
    '''Both access forms of an earthaccess granule's data link.

    Direct-S3 links only open in-region (the Lambda fleet, us-west-2);
    off-region local streaming needs the HTTPS form — earthaccess refuses
    S3 links outside the region.
    '''
    urls = {}
    direct = granule.data_links(access='direct')
    if direct:
        urls['s3'] = direct[0]
    external = granule.data_links(access='external') or granule.data_links()
    if external:
        urls['https'] = external[0]
    if not urls:
        raise ValueError(f'no data links found for granule: {granule}')
    return urls


def build_granule_index(time_range, collections=None, cache_path=None):
    '''
    Query earthaccess for every MERRA-2 granule the catalog span needs.

    Queries each unique DOI once (merra2_flx/merra2_precip share files) and
    fans the result out to each collection name.

    Inputs:
        time_range (tuple of str): (start, end) as YYYY-MM-DD, e.g. the
            catalog's (t_start.min(), t_end_augmented.max())
        collections (list of str): collection names to index; defaults to
            every key in MERRA2_DOIS
        cache_path (str or Path): JSON cache; loaded instead of re-querying
            when it covers the requested range and collections. The cache is
            written as {"meta": {"time_range": [start, end], "collections": [...]},
            "index": {...}} so a later call with a different range/collections
            can tell whether the cached content actually satisfies it.

    Outputs:
        index (dictionary): {collection_name: {date: url}}
    '''
    if collections is None:
        collections = list(MERRA2_DOIS)

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with open(cache_path) as f:
                cached = json.load(f)
            meta = cached.get('meta') if isinstance(cached, dict) else None
            if meta is None or meta.get('version') != _CACHE_VERSION:
                logger.info(
                    'cache %s has no meta or an older layout; re-querying', cache_path
                )
            else:
                cached_start, cached_end = meta['time_range']
                cached_collections = set(meta['collections'])
                covers_range = cached_start <= time_range[0] and cached_end >= time_range[1]
                covers_collections = cached_collections.issuperset(collections)
                if covers_range and covers_collections:
                    logger.info('loading cached granule index from %s', cache_path)
                    return {name: cached['index'][name] for name in collections}
                logger.info(
                    'cache %s does not cover requested range/collections '
                    '(cached range=%s collections=%s; requested range=%s '
                    'collections=%s); re-querying',
                    cache_path,
                    meta['time_range'],
                    sorted(cached_collections),
                    list(time_range),
                    list(collections),
                )

    import earthaccess

    earthaccess.login()
    by_doi = {}
    for name in collections:
        doi = MERRA2_DOIS[name]
        if doi in by_doi:
            continue
        logger.info('querying earthaccess for %s (%s) over %s', name, doi, time_range)
        granules = earthaccess.search_data(doi=doi, temporal=time_range, count=-1)
        logger.info('  %d granules', len(granules))
        dates = {}
        for granule in granules:
            try:
                dates[_granule_date(granule)] = _granule_urls(granule)
            except ValueError as e:
                logger.warning('skipping granule: %s', e)
        by_doi[doi] = dates

    index = {name: by_doi[MERRA2_DOIS[name]] for name in collections}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'meta': {
                'version': _CACHE_VERSION,
                'time_range': list(time_range),
                'collections': list(collections),
            },
            'index': index,
        }
        with open(cache_path, 'w') as f:
            json.dump(payload, f)
        logger.info('cached granule index to %s', cache_path)
    return index


def granule_urls_for_range(index, collection, t_start, t_end, *, access='s3'):
    '''
    The granule URLs covering [t_start, t_end] for one collection.

    MERRA-2 granules are daily files, so this is one URL per calendar day
    touched by the range. A day missing from the index raises KeyError (a
    silent gap would surface as a wrong attribute, not an error).

    Inputs:
        index (dictionary): from build_granule_index
        collection (string): collection name
        t_start, t_end: timestamps (anything pd.Timestamp accepts)
        access (string): 's3' (direct links, in-region/Lambda) or 'https'
            (off-region local streaming)

    Outputs:
        urls (list of str)
    '''
    dates = pd.date_range(
        pd.Timestamp(t_start).floor('D'), pd.Timestamp(t_end).floor('D'), freq='D'
    )
    by_date = index[collection]
    missing = [d.strftime('%Y-%m-%d') for d in dates if d.strftime('%Y-%m-%d') not in by_date]
    if missing:
        raise KeyError(
            f'granule index for {collection!r} is missing {len(missing)} day(s) '
            f'in [{t_start}, {t_end}]: {missing[:5]}...'
            if len(missing) > 5
            else f'granule index for {collection!r} is missing day(s): {missing}'
        )
    urls = []
    for d in dates:
        forms = by_date[d.strftime('%Y-%m-%d')]
        if access not in forms:
            raise KeyError(
                f'granule index for {collection!r} has no {access!r} link for '
                f'{d.date()} (available: {sorted(forms)})'
            )
        urls.append(forms[access])
    return urls


def _event_time_range(row, augmented):
    end = row['t_end_augmented'] if augmented else row['t_end']
    return row['t_start'], end


def events_for_zagg(
    events, index, collections, static_uris, *, augmented=False, uri_map=None
):
    '''
    Build the Lambda-backend URI dicts for a set of exported storms.

    One dict per storm, per zagg's event contract (zagg
    docs/temporal_events.md): the mask travels by URI, each requested
    collection maps to the granule URLs covering the storm's time range
    (the *augmented* range when aggregating under the augmented masks).

    Inputs:
        events (pd.DataFrame): rows from zagg_events.export_events
        index (dictionary): from build_granule_index
        collections (list of str): collection names this run reads
        static_uris (dictionary): {static_name: uri} forwarded verbatim
        augmented (boolean): use the augmented masks + time ranges (precip)
        uri_map (callable): optional local-path -> URI translation (e.g. the
            mapping returned by mirror_to_s3); default uses local paths, which
            only works for local smoke tests of the payload shape

    Outputs:
        payloads (list of dict)
    '''
    mask_col = 'mask_augmented_path' if augmented else 'mask_path'
    translate = uri_map or (lambda p: p)
    payloads = []
    for _, row in events.iterrows():
        t_start, t_end = _event_time_range(row, augmented)
        local_mask = row[mask_col]
        mask_uri = translate(local_mask)
        if not mask_uri:
            raise ValueError(
                f'no mask URI for event {row["event_key"]!r}: local path '
                f'{local_mask!r} is not in the uri_map. Ensure mirror_to_s3 '
                f'uploaded it (augmented_too=True for the augmented masks).'
            )
        payloads.append(
            {
                'event_key': row['event_key'],
                'event_mask_uri': mask_uri,
                'collection_uris': {
                    name: granule_urls_for_range(index, name, t_start, t_end)
                    for name in collections
                },
                'static_uris': dict(static_uris),
            }
        )
    return payloads


def local_event_tuples(events, index, collections, static_data, *, augmented=False):
    '''
    Build the local-backend event tuples for a set of exported storms.

    Opens each storm's granules through earthaccess (the same streaming path
    compute_attributes_streaming uses) as lazy multi-file datasets; zagg's
    worker subsets to the event extent and applies the config's per-collection
    options (time_offset/resample/derived), so the datasets are passed raw.
    Coordinates are rounded to 5 decimals, matching the loading_utils
    convention, so `.sel` against the mask coordinates matches exactly.

    Inputs:
        events (pd.DataFrame): rows from zagg_events.export_events
        index (dictionary): from build_granule_index
        collections (list of str): collection names this run reads
        static_data (dictionary): {name: DataArray/Dataset}, already loaded
            (e.g. via load_zagg_statics)
        augmented (boolean): use the augmented masks + time ranges (precip)

    Outputs:
        tuples (list): (event_key, event_mask, collections, static_data)
            entries for zagg.agg(events=...)
    '''
    import earthaccess
    import xarray as xr

    mask_col = 'mask_augmented_path' if augmented else 'mask_path'
    earthaccess.login()
    tuples = []
    for _, row in events.iterrows():
        t_start, t_end = _event_time_range(row, augmented)
        mask = xr.open_dataarray(row[mask_col])
        # export_events now rounds mask coords at write time, so this is a cheap
        # no-op safety net (harmless for masks exported before that change).
        mask = mask.assign_coords(lat=mask.lat.round(5), lon=mask.lon.round(5))
        opened = {}
        for name in collections:
            # https form: direct-S3 links only open in-region (Lambda)
            urls = granule_urls_for_range(index, name, t_start, t_end, access='https')
            ds = xr.open_mfdataset(earthaccess.open(urls))
            opened[name] = ds.assign_coords(lat=ds.lat.round(5), lon=ds.lon.round(5))
        tuples.append((row['event_key'], mask, opened, static_data))
    return tuples


def mirror_to_s3(events, static_paths, s3_prefix, *, augmented_too=True):
    '''
    Upload exported masks and statics to S3 for a Lambda fan-out.

    Inputs:
        events (pd.DataFrame): rows from zagg_events.export_events
        static_paths (dictionary): {static_name: local path}
        s3_prefix (string): e.g. 's3://bucket/ar-events/eps0.5_.../'
        augmented_too (boolean): also mirror the augmented masks

    Outputs:
        uri_map (dictionary): {local_path: s3_uri} for every uploaded file
            (pass uri_map.get into events_for_zagg)
        static_uris (dictionary): {static_name: s3_uri}
    '''
    import boto3

    if not s3_prefix.startswith('s3://'):
        raise ValueError(f's3_prefix must be an s3:// URI, got {s3_prefix!r}')
    bucket, _, prefix = s3_prefix[5:].partition('/')
    prefix = prefix.rstrip('/')
    client = boto3.client('s3')

    def _upload(local_path, subdir):
        local_path = Path(local_path)
        key = f'{prefix}/{subdir}/{local_path.name}'
        client.upload_file(str(local_path), bucket, key)
        return f's3://{bucket}/{key}'

    uri_map = {}
    for _, row in events.iterrows():
        uri_map[row['mask_path']] = _upload(row['mask_path'], 'masks')
        if augmented_too:
            uri_map[row['mask_augmented_path']] = _upload(
                row['mask_augmented_path'], 'masks_augmented'
            )
    static_uris = {}
    for name, path in static_paths.items():
        uri = _upload(path, 'static')
        static_uris[name] = uri
        uri_map[str(path)] = uri
    return uri_map, static_uris
