# Shared settings for the PBS jobs. Sourced by every *.pbs script.
#
# Adjust the module names to what `module avail` shows on Vanda, and the paths
# to where you cloned the repository and prepared the data.

module purge 2>/dev/null || true
module load gcc openmpi cmake 2>/dev/null || true
if [[ "${NEED_CUDA:-0}" == "1" ]]; then
  module load cuda 2>/dev/null || true
fi

ROOT=${ROOT:-$PBS_O_WORKDIR}
BUILD=${BUILD:-$ROOT/build}
DATA=${DATA:-$ROOT/data}
RESULTS=${RESULTS:-$ROOT/results}
TRAIN=${TRAIN:-$DATA/higgs_train.bin}
TEST=${TEST:-$DATA/higgs_test.bin}
mkdir -p "$RESULTS"
cd "$ROOT"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Number of cores per node allocated to the job (from the PBS select line).
CORES_PER_NODE=${CORES_PER_NODE:-$(nproc)}
N_NODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l)
[[ "$N_NODES" -ge 1 ]] || N_NODES=1

# run_mpi RANKS THREADS ARGS...  — places RANKS ranks evenly over the nodes,
# each bound to THREADS consecutive cores (Open MPI syntax).
run_mpi() {
  local ranks=$1 threads=$2
  shift 2
  local nodes=$N_NODES
  (( ranks < nodes )) && nodes=$ranks
  local per_node=$(( (ranks + nodes - 1) / nodes ))
  OMP_NUM_THREADS=$threads mpirun -np "$ranks" \
    --map-by "ppr:${per_node}:node:PE=${threads}" --bind-to core \
    -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
    "$BUILD/dt_mpi" "$@"
}
