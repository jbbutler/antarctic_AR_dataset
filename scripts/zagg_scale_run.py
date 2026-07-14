'''
Operator scale run: the full AR catalog through zagg's Lambda fan-out.

The phase-G runner (englacial/zagg#213): exports storms as zagg events,
mirrors masks + statics to S3, resolves granule URLs, and dispatches one
``process_event`` Lambda invoke per storm — the raw-mask attribute run and
the augmented-mask precip run — collecting both Parquet outputs under the
given S3 prefix. Requires a standing zagg Lambda stack (us-west-2 — GES DISC
S3 is in-region only), an S3 bucket you can write, an Earthdata login
(~/.netrc), and AWS credentials for the invoking profile. This spends real
money; the default is a 10-storm slice — pass --full for all storms, ideally
after the slice reproduces scripts/zagg_parity.py's numbers for the same
storms.

Usage:
    AWS_PROFILE=... python scripts/zagg_scale_run.py \
        --s3-prefix s3://YOUR-BUCKET/ar-events/eps0.5-seed12345 \
        [--catalog epsspace0.5_..._seed12345.h5] [--n-storms 10 | --full]
        [--function-name process-shard] [--max-workers 1000] [--skip-mirror]

Jimmy Butler & co.
July 2026
'''

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from artools.zagg_events import export_events  # noqa: E402
from artools.zagg_granules import (  # noqa: E402
    build_granule_index,
    events_for_zagg,
    mirror_to_s3,
)
from artools.zagg_statics import prepare_static_files  # noqa: E402

logger = logging.getLogger('zagg_scale_run')

DEFAULT_CATALOG = 'epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5'
FIELD_COLLECTIONS = ['merra2_slv', 'merra2_flx', 'merra2_v850', 'merra2_omega']


def _mirror(events, static_paths, s3_prefix, cache_path, skip):
    '''Mirror masks/statics to S3, or reload a previous run's URI maps.'''
    if skip:
        if not cache_path.exists():
            raise FileNotFoundError(
                f'--skip-mirror needs a prior mirror map at {cache_path}'
            )
        cached = json.loads(cache_path.read_text())
        # the cache is prefix-agnostic, so a --skip-mirror against a different
        # prefix would read inputs from the mirrored one while writing outputs
        # elsewhere; refuse unless the cached prefix matches (a legacy cache
        # with no recorded prefix counts as a mismatch)
        cached_prefix = cached.get('s3_prefix')
        if cached_prefix != s3_prefix:
            raise ValueError(
                f'cached mirror map targets prefix {cached_prefix!r} but this run '
                f'requested {s3_prefix!r}; --skip-mirror would read inputs from the '
                f'cached prefix while writing outputs to the requested one -- re-run '
                f'without --skip-mirror to mirror into {s3_prefix!r}'
            )
        missing = [p for p in events['mask_path'] if p not in cached['uri_map']]
        missing += [
            p for p in events['mask_augmented_path'] if p not in cached['uri_map']
        ]
        if missing:
            raise KeyError(
                f'{len(missing)} mask file(s) are not in the cached mirror map '
                f'(first: {missing[0]!r}); re-run without --skip-mirror'
            )
        # statics aren't overwrite-guarded like masks: a cache written before the
        # climatology existed would omit it (anomalies fan out baseline-less), and
        # a rebuilt static leaves the cache pointing at a stale S3 object
        static_uris = cached['static_uris']
        static_mtimes = cached.get('static_mtimes', {})
        missing_statics = [name for name in static_paths if name not in static_uris]
        if missing_statics:
            raise KeyError(
                f'static file(s) {missing_statics} are not in the cached mirror map; '
                f're-run without --skip-mirror to mirror them'
            )
        stale = [
            name for name, path in static_paths.items()
            if Path(path).stat().st_mtime > static_mtimes.get(name, 0.0)
        ]
        if stale:
            raise ValueError(
                f'static file(s) {stale} are newer than the cached mirror; the '
                f'mirrored copies are stale -- re-run without --skip-mirror'
            )
        return cached['uri_map'], static_uris

    logger.info('mirroring %d storms (raw + augmented) + statics to %s',
                len(events), s3_prefix)
    uri_map, static_uris = mirror_to_s3(events, static_paths, s3_prefix)
    static_mtimes = {name: Path(path).stat().st_mtime for name, path in static_paths.items()}
    cache_path.write_text(json.dumps({
        's3_prefix': s3_prefix,
        'uri_map': uri_map,
        'static_uris': static_uris,
        'static_mtimes': static_mtimes,
    }))
    logger.info('mirror map cached at %s', cache_path)
    return uri_map, static_uris


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    import zagg
    from zagg.config import load_config

    catalog_path = Path(ARGS.catalogs_dir) / ARGS.catalog

    if ARGS.full:
        storms = None
        logger.info('FULL catalog run')
    else:
        catalog = pd.read_hdf(catalog_path)
        storms = [int(i) for i in catalog.index[:ARGS.n_storms]]
        logger.info('slice run: %d storms', len(storms))

    events, _ = export_events(catalog_path, storms=storms)
    static_paths = prepare_static_files(ARGS.catalogs_dir)
    if 'climatology' not in static_paths:
        raise FileNotFoundError(
            'no monthly climatology under <catalogs dir>/static/ -- build it '
            'first (artools.zagg_statics.build_climatology) with the baseline '
            'the science calls for; the anomaly attributes need it'
        )

    span = (
        str(pd.Timestamp(events['t_start'].min()).date()),
        str(pd.Timestamp(events['t_end_augmented'].max()).date()),
    )
    index = build_granule_index(
        span, cache_path=Path(ARGS.catalogs_dir) / 'derived' / 'granule_index.json'
    )

    prefix = ARGS.s3_prefix.rstrip('/')
    mirror_cache = catalog_path.parent / 'derived' / catalog_path.stem / 'mirror_map.json'
    uri_map, static_uris = _mirror(
        events, static_paths, prefix, mirror_cache, ARGS.skip_mirror
    )

    runs = [
        ('configs/ar_attributes.yaml', FIELD_COLLECTIONS, False,
         f'{prefix}/output/ar_attributes.parquet'),
        ('configs/ar_precip.yaml', ['merra2_precip'], True,
         f'{prefix}/output/ar_precip.parquet'),
    ]
    summaries = {}
    for config_path, collections, augmented, store in runs:
        payloads = events_for_zagg(
            events, index, collections, static_uris,
            augmented=augmented, uri_map=uri_map.get,
        )
        logger.info('dispatching %d events -> %s', len(payloads), store)
        summary = zagg.agg(
            load_config(REPO / config_path),
            events=payloads,
            backend='lambda',
            store=store,
            function_name=ARGS.function_name,
            region=ARGS.region,
            max_workers=ARGS.max_workers,
        )
        summaries[config_path] = summary
        logger.info(
            '%s: %d ok, %d errors, %.1fs wall -> %s',
            config_path, summary['events_with_data'], summary['events_error'],
            summary['wall_time_s'], summary.get('output_path'),
        )

    failed = sum(s['events_error'] for s in summaries.values())
    print('\n== scale run summary ==')
    for name, s in summaries.items():
        print(f'{name}: {s["events_with_data"]}/{s["total_events"]} events, '
              f'{s["events_error"]} errors, {s["wall_time_s"]:.0f}s '
              f'-> {s.get("output_path")}')
    if failed:
        print(f'{failed} event(s) failed -- rerun their storms or inspect the '
              f'status objects next to the store paths')
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--s3-prefix', required=True,
                        help='s3://bucket/prefix for mirrored inputs + outputs')
    parser.add_argument('--catalog', default=DEFAULT_CATALOG)
    parser.add_argument('--catalogs-dir', default=str(REPO.parent / 'antarctic_AR_catalogs'))
    parser.add_argument('--n-storms', type=int, default=10,
                        help='slice size (ignored with --full)')
    parser.add_argument('--full', action='store_true',
                        help='run every storm in the catalog')
    parser.add_argument('--function-name', default=None,
                        help='zagg Lambda function (default: ZAGG_LAMBDA_FUNCTION_NAME env)')
    parser.add_argument('--region', default='us-west-2')
    parser.add_argument('--max-workers', type=int, default=1000)
    parser.add_argument('--skip-mirror', action='store_true',
                        help='reuse the previous mirror map instead of re-uploading')
    ARGS = parser.parse_args()
    sys.exit(main())
