import json

import numpy as np
import pandas as pd
import pytest

from artools.zagg_granules import (
    MERRA2_DOIS,
    build_granule_index,
    events_for_zagg,
    granule_urls_for_range,
    mirror_to_s3,
)


def _index(days=('1980-01-02', '1980-01-03', '1980-01-04')):
    return {
        name: {d: f's3://gesdisc/{name}/{d}.nc4' for d in days}
        for name in ('merra2_slv', 'merra2_precip')
    }


def _events():
    return pd.DataFrame(
        [
            {
                'event_key': 'storm_00001',
                'mask_path': '/local/masks/storm_00001.nc',
                'mask_augmented_path': '/local/masks_augmented/storm_00001.nc',
                't_start': np.datetime64('1980-01-02T00'),
                't_end': np.datetime64('1980-01-03T18'),
                't_end_augmented': np.datetime64('1980-01-04T18'),
            }
        ]
    )


class _FakeGranule(dict):
    def __init__(self, date, url):
        super().__init__(
            umm={'TemporalExtent': {'RangeDateTime': {'BeginningDateTime': f'{date}T00:00:00Z'}}}
        )
        self._url = url

    def data_links(self, access=None):
        return [self._url]


def test_build_granule_index_queries_each_doi_once(monkeypatch, tmp_path):
    import earthaccess

    queried = []

    def _search(doi, temporal, count):
        queried.append(doi)
        return [_FakeGranule('1980-01-02', f's3://x/{doi}/a.nc4')]

    monkeypatch.setattr(earthaccess, 'login', lambda: None)
    monkeypatch.setattr(earthaccess, 'search_data', _search)

    cache = tmp_path / 'index.json'
    index = build_granule_index(('1980-01-01', '1980-01-05'), cache_path=cache)

    # flx and precip share a DOI: 5 collections, 4 queries
    assert len(queried) == len(set(queried)) == 4
    assert index['merra2_flx'] == index['merra2_precip']
    assert index['merra2_slv'] == {'1980-01-02': f's3://x/{MERRA2_DOIS["merra2_slv"]}/a.nc4'}
    # cache round-trip short-circuits the query
    queried.clear()
    again = build_granule_index(('1980-01-01', '1980-01-05'), cache_path=cache)
    assert queried == []
    assert again == json.loads(cache.read_text())['index']
    # cache is written with metadata alongside the index
    cached = json.loads(cache.read_text())
    assert cached['meta']['time_range'] == ['1980-01-01', '1980-01-05']
    assert set(cached['meta']['collections']) == set(MERRA2_DOIS)


def _seed_cache(monkeypatch, cache, time_range, collections, queried):
    import earthaccess

    def _search(doi, temporal, count):
        queried.append(doi)
        return [_FakeGranule('1980-01-02', f's3://x/{doi}/a.nc4')]

    monkeypatch.setattr(earthaccess, 'login', lambda: None)
    monkeypatch.setattr(earthaccess, 'search_data', _search)
    build_granule_index(time_range, collections=collections, cache_path=cache)


def test_build_granule_index_subset_collections_serves_from_cache(monkeypatch, tmp_path):
    cache = tmp_path / 'index.json'
    queried = []
    _seed_cache(monkeypatch, cache, ('1980-01-01', '1980-01-05'), None, queried)

    # a subset of the cached collections is served without re-querying
    queried.clear()
    index = build_granule_index(
        ('1980-01-01', '1980-01-05'), collections=['merra2_slv'], cache_path=cache
    )
    assert queried == []
    assert set(index) == {'merra2_slv'}


def test_build_granule_index_superset_collections_requeries(monkeypatch, tmp_path):
    cache = tmp_path / 'index.json'
    queried = []
    # seed a cache holding only one collection
    _seed_cache(monkeypatch, cache, ('1980-01-01', '1980-01-05'), ['merra2_slv'], queried)

    # requesting a collection the cache lacks forces a re-query
    queried.clear()
    index = build_granule_index(
        ('1980-01-01', '1980-01-05'),
        collections=['merra2_slv', 'merra2_v850'],
        cache_path=cache,
    )
    assert queried  # re-queried rather than KeyError on the missing collection
    assert set(index) == {'merra2_slv', 'merra2_v850'}


def test_build_granule_index_narrower_cached_range_requeries(monkeypatch, tmp_path):
    cache = tmp_path / 'index.json'
    queried = []
    _seed_cache(monkeypatch, cache, ('1980-01-02', '1980-01-03'), None, queried)

    # a wider requested range is not covered by the narrower cache
    queried.clear()
    build_granule_index(('1980-01-01', '1980-01-05'), cache_path=cache)
    assert queried


def test_build_granule_index_legacy_cache_requeries(monkeypatch, tmp_path):
    import earthaccess

    cache = tmp_path / 'index.json'
    cache.write_text(json.dumps({'merra2_slv': {'1980-01-02': 's3://x/a.nc4'}}))

    queried = []

    def _search(doi, temporal, count):
        queried.append(doi)
        return [_FakeGranule('1980-01-02', f's3://x/{doi}/a.nc4')]

    monkeypatch.setattr(earthaccess, 'login', lambda: None)
    monkeypatch.setattr(earthaccess, 'search_data', _search)

    build_granule_index(('1980-01-01', '1980-01-05'), cache_path=cache)
    assert queried  # a meta-less legacy cache is a miss


def test_granule_urls_for_range_one_per_day():
    urls = granule_urls_for_range(
        _index(), 'merra2_slv', np.datetime64('1980-01-02T06'), np.datetime64('1980-01-03T18')
    )
    assert urls == [
        's3://gesdisc/merra2_slv/1980-01-02.nc4',
        's3://gesdisc/merra2_slv/1980-01-03.nc4',
    ]


def test_granule_urls_missing_day_raises():
    with pytest.raises(KeyError, match='1980-01-05'):
        granule_urls_for_range(
            _index(), 'merra2_slv', np.datetime64('1980-01-04'), np.datetime64('1980-01-05T18')
        )


def test_events_for_zagg_raw_and_augmented_ranges():
    events = _events()
    statics = {'ais_mask': 's3://b/static/ais.nc'}

    raw = events_for_zagg(events, _index(), ['merra2_slv'], statics)
    (payload,) = raw
    assert payload['event_key'] == 'storm_00001'
    assert payload['event_mask_uri'] == '/local/masks/storm_00001.nc'
    assert len(payload['collection_uris']['merra2_slv']) == 2  # Jan 2-3
    assert payload['static_uris'] == statics

    aug = events_for_zagg(events, _index(), ['merra2_precip'], statics, augmented=True)
    (payload,) = aug
    assert payload['event_mask_uri'] == '/local/masks_augmented/storm_00001.nc'
    assert len(payload['collection_uris']['merra2_precip']) == 3  # Jan 2-4


def test_events_for_zagg_uri_map_translates_masks():
    uri_map = {'/local/masks/storm_00001.nc': 's3://b/masks/storm_00001.nc'}
    (payload,) = events_for_zagg(
        _events(), _index(), ['merra2_slv'], {}, uri_map=uri_map.get
    )
    assert payload['event_mask_uri'] == 's3://b/masks/storm_00001.nc'


def test_events_for_zagg_unmapped_mask_raises():
    # an empty uri_map leaves the mask path untranslated -> fail fast
    with pytest.raises(ValueError, match='storm_00001'):
        events_for_zagg(_events(), _index(), ['merra2_slv'], {}, uri_map={}.get)


def test_mirror_to_s3_uploads_and_maps(monkeypatch, tmp_path):
    boto3 = pytest.importorskip('boto3')

    uploads = []

    class _FakeClient:
        def upload_file(self, path, bucket, key):
            uploads.append((path, bucket, key))

    monkeypatch.setattr(boto3, 'client', lambda service: _FakeClient())

    mask = tmp_path / 'storm_00001.nc'
    aug = tmp_path / 'storm_00001_aug.nc'
    ais = tmp_path / 'ais.nc'
    for f in (mask, aug, ais):
        f.write_bytes(b'x')

    events = _events()
    events.loc[0, 'mask_path'] = str(mask)
    events.loc[0, 'mask_augmented_path'] = str(aug)

    uri_map, static_uris = mirror_to_s3(events, {'ais_mask': ais}, 's3://bkt/run1/')

    assert uri_map[str(mask)] == 's3://bkt/run1/masks/storm_00001.nc'
    assert uri_map[str(aug)] == 's3://bkt/run1/masks_augmented/storm_00001_aug.nc'
    assert static_uris == {'ais_mask': 's3://bkt/run1/static/ais.nc'}
    assert {b for _, b, _ in uploads} == {'bkt'}
    assert len(uploads) == 3


def test_mirror_to_s3_rejects_non_s3_prefix():
    with pytest.raises(ValueError, match='s3://'):
        mirror_to_s3(_events(), {}, '/not/s3')
