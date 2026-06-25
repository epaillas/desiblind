#!/usr/bin/env bash
#SBATCH --job-name=desiblind-bao-lss-validate
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out

# Submit from the desiblind checkout, e.g.:
#   sbatch scripts/validation/submit_catalog_bao_lss_split_gc_validation.sh
# or for several random files:
#   sbatch --array=0-17 scripts/validation/submit_catalog_bao_lss_split_gc_validation.sh
#
# Environment variables:
#   RANNUM              Random number to validate; defaults to SLURM_ARRAY_TASK_ID or 1.
#   NROWS               Data rows to use; 0 means full file. Default: 0.
#   RANDOM_NROWS        Random rows to use; 0 means full random. Default: 0.
#   TRACER              LSS tracer type. Default: LRG.
#   TRACER_NAME         desiblind canonical tracer-bin name. Default: LRG3.
#   VERSION_DIR         LSS catalog version directory. Default: DA2 loa-v1 v2.1 path.
#   NZ_RANDOM_CATALOG   Random catalog for mknz_full area normalization. Default: ${VERSION_DIR}/${TRACER}_1_full_HPmapcut.ran.fits.
#   RANDOM_CATALOG      Random catalog processed by mkclusran. Default: ${VERSION_DIR}/${TRACER}_${RANNUM}_full_HPmapcut.ran.fits.
#   REAL_CATALOG        Full data catalog. Default: ${VERSION_DIR}/${TRACER}_full_HPmapcut.dat.fits.
#   OUTPUT_DIR          Output directory. Default: $SCRATCH/desiblind_lss_validation/production-${SLURM_JOB_ID}-${RANNUM}.
# Extra command-line arguments are forwarded to validate_catalog_bao_lss_split_gc.py.

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"

set +u
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

RANNUM=${RANNUM:-${SLURM_ARRAY_TASK_ID:-1}}
NROWS=${NROWS:-0}
RANDOM_NROWS=${RANDOM_NROWS:-0}
TRACER=${TRACER:-LRG}
TRACER_NAME=${TRACER_NAME:-LRG3}
VERSION_DIR=${VERSION_DIR:-/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1}
REAL_CATALOG=${REAL_CATALOG:-${VERSION_DIR}/${TRACER}_full_HPmapcut.dat.fits}
NZ_RANDOM_CATALOG=${NZ_RANDOM_CATALOG:-${VERSION_DIR}/${TRACER}_1_full_HPmapcut.ran.fits}
RANDOM_CATALOG=${RANDOM_CATALOG:-${VERSION_DIR}/${TRACER}_${RANNUM}_full_HPmapcut.ran.fits}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRATCH:-.}/desiblind_lss_validation/production-${SLURM_JOB_ID:-manual}-${RANNUM}}

python scripts/validation/validate_catalog_bao_lss_split_gc.py   --real-catalog "$REAL_CATALOG"   --nz-random-catalog "$NZ_RANDOM_CATALOG"   --random-catalog "$RANDOM_CATALOG"   --rannum "$RANNUM"   --nrows "$NROWS"   --random-nrows "$RANDOM_NROWS"   --tracer-type "$TRACER"   --tracer-name "$TRACER_NAME"   --output-dir "$OUTPUT_DIR"   "$@"
