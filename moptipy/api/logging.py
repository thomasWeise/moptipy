"""Shared constants and functions for dealing with logs."""

from typing import Final

#: the file suffix to be used for log files
FILE_SUFFIX: Final[str] = ".txt"

#: the key for the exception type
KEY_EXCEPTION_TYPE: Final[str] = "exceptionType"
#: the key for the exception value
KEY_EXCEPTION_VALUE: Final[str] = "exceptionValue"
#: the key for the exception stack trace
KEY_EXCEPTION_STACK_TRACE: Final[str] = "exceptionStackTrace"

#: the key for algorithms
KEY_ALGORITHM: Final[str] = "algorithm"
#: the key for the instance
KEY_INSTANCE: Final[str] = "instance"

#: the common name prefix of all error sections
ERROR_SECTION_PREFIX: Final[str] = "ERROR_"
#: the section indicating an error during the algorithm run
SECTION_ERROR_IN_RUN: Final[str] = f"{ERROR_SECTION_PREFIX}IN_RUN"
#: the section indicating an error that occurred in the context of the
#: process. this may be an error in the algorithm or, more likely, in the
#: processing of the result.
SECTION_ERROR_IN_CONTEXT: Final[str] = f"{ERROR_SECTION_PREFIX}IN_CONTEXT"
#: the section indicating an invalid candidate solution
SECTION_ERROR_INVALID_Y: Final[str] = f"{ERROR_SECTION_PREFIX}INVALID_Y"
#: the section indicating a mismatch of the computed and registered best f
SECTION_ERROR_BEST_F: Final[str] = f"{ERROR_SECTION_PREFIX}BEST_F_MISMATCH"
#: the section indicating an invalid point in the search space
SECTION_ERROR_INVALID_X: Final[str] = f"{ERROR_SECTION_PREFIX}INVALID_X"
#: the section indicating that the time measurement has an error
SECTION_ERROR_TIMING: Final[str] = f"{ERROR_SECTION_PREFIX}TIMING"
#: the section indicating an error caught during log writing
SECTION_ERROR_IN_LOG: Final[str] = f"{ERROR_SECTION_PREFIX}IN_LOG"

#: the progress csv section
SECTION_PROGRESS: Final[str] = "PROGRESS"
#: the FEs column for the progress CSV
PROGRESS_FES: Final[str] = "fes"
#: the time millis column for the progress CSV
PROGRESS_TIME_MILLIS: Final[str] = "timeMS"
#: the current objective value column for the progress CSV
PROGRESS_CURRENT_F: Final[str] = "f"

#: the end state
SECTION_FINAL_STATE: Final[str] = "STATE"
#: the total number of consumed FEs
KEY_TOTAL_FES: Final[str] = "totalFEs"
#: the total number of consumed milliseconds
KEY_TOTAL_TIME_MILLIS: Final[str] = "totalTimeMillis"
#: the best encountered objective value
KEY_BEST_F: Final[str] = "bestF"
#: the vector of objective values of the best solution
KEY_BEST_FS: Final[str] = "bestFs"
#: the number of archived non-dominated solutions
KEY_ARCHIVE_SIZE: Final[str] = "archiveSize"
#: the FE when the best objective value was reached
KEY_LAST_IMPROVEMENT_FE: Final[str] = "lastImprovementFE"
#: the time in milliseconds when the best objective value was reached
KEY_LAST_IMPROVEMENT_TIME_MILLIS: Final[str] = "lastImprovementTimeMillis"

#: the archive section prefix
PREFIX_SECTION_ARCHIVE: Final[str] = "ARCHIVE_"
#: the archive quality section suffix
SECTION_ARCHIVE_QUALITY: Final[str] = f"{PREFIX_SECTION_ARCHIVE}QUALITIES"
#: the scalarized objective value column of the archive if without numeric
#: suffix, the result of the ith objective function if with numeric suffix
KEY_ARCHIVE_F: Final[str] = PROGRESS_CURRENT_F

#: the archive X section suffix
SUFFIX_SECTION_ARCHIVE_X: Final[str] = "_X"
#: the archive Y section suffix
SUFFIX_SECTION_ARCHIVE_Y: Final[str] = "_Y"

#: the setup section
SECTION_SETUP: Final[str] = "SETUP"
#: the default log key for names of objects
KEY_NAME: Final[str] = "name"
#: the class of an object
KEY_CLASS: Final[str] = "class"
#: the inner class of an object
KEY_INNER_CLASS: Final[str] = "innerClass"
#: the default log key for the lower bound of objective function values
KEY_F_LOWER_BOUND: Final[str] = "lowerBound"
#: the default log key for the upper bound of objective function values
KEY_F_UPPER_BOUND: Final[str] = "upperBound"
#: the maximum FEs of a black-box process
KEY_MAX_FES: Final[str] = "maxFEs"
#: the maximum runtime in milliseconds of a black-box process
KEY_MAX_TIME_MILLIS: Final[str] = "maxTimeMillis"
#: the goal objective value of a black-box process
KEY_GOAL_F: Final[str] = "goalF"
#: the random seed
KEY_RAND_SEED: Final[str] = "randSeed"
#: the random generator type
KEY_RAND_GENERATOR_TYPE: Final[str] = "randGenType"
#: the type of the bit generator used by the random generator
KEY_RAND_BIT_GENERATOR_TYPE: Final[str] = "randBitGenType"
#: the number of decision variables
KEY_SPACE_NUM_VARS: Final[str] = "nvars"
#: the maximum archive size (after pruning)
KEY_ARCHIVE_MAX_SIZE: Final[str] = "archiveMaxSize"
#: the pruning limit of the archive size
KEY_ARCHIVE_PRUNE_LIMIT: Final[str] = "archivePruneLimit"

#: the scope of the process parameters
SCOPE_PROCESS: Final[str] = "p"
#: the scope of the solution space
SCOPE_SOLUTION_SPACE: Final[str] = "y"
#: the scope of the search space
SCOPE_SEARCH_SPACE: Final[str] = "x"
#: the scope of the objective function
SCOPE_OBJECTIVE_FUNCTION: Final[str] = "f"
#: the scope of the encoding
SCOPE_ENCODING: Final[str] = "g"
#: the scope of the optimization algorithm
SCOPE_ALGORITHM: Final[str] = "a"
#: the scope of the multi-objective archive pruner
SCOPE_PRUNER: Final[str] = "ap"
#: the scope of the nullary search operator
SCOPE_OP0: Final[str] = "op0"
#: the scope of the unary search operator
SCOPE_OP1: Final[str] = "op1"
#: the scope of the binary search operator
SCOPE_OP2: Final[str] = "op2"

#: the automatically generated system info section
SECTION_SYS_INFO: Final[str] = "SYS_INFO"
#: information about the current session
SCOPE_SESSION: Final[str] = "session"
#: the time when the session was started
KEY_SESSION_START: Final[str] = "start"
#: the node on which the session is running
KEY_NODE_NAME: Final[str] = "node"
#: the affinity of the process to logical CPUs
KEY_CPU_AFFINITY: Final[str] = "cpuAffinity"
#: the pid of the process
KEY_PROCESS_ID: Final[str] = "processId"
#: the command line of the process
KEY_COMMAND_LINE: Final[str] = "commandLine"
#: the working directory of the process
KEY_WORKING_DIRECTORY: Final[str] = "workingDirectory"
#: the ip address of the node on which the session is running
KEY_NODE_IP: Final[str] = "ipAddress"
#: the versions scope in the sys-info section
SCOPE_VERSIONS: Final[str] = "version"
#: the hardware scope in the sys-info section
SCOPE_HARDWARE: Final[str] = "hardware"
#: the number of physical CPUs
KEY_HW_N_PHYSICAL_CPUS: Final[str] = "nPhysicalCpus"
#: the number of logical CPUs
KEY_HW_N_LOGICAL_CPUS: Final[str] = "nLogicalCpus"
#: the clock speed of the CPUs
KEY_HW_CPU_MHZ: Final[str] = "cpuMhz"
#: the key for the byte order
KEY_HW_BYTE_ORDER: Final[str] = "byteOrder"
#: the key for the machine
KEY_HW_MACHINE: Final[str] = "machine"
#: the key for the cpu name
KEY_HW_CPU_NAME: Final[str] = "cpu"
#: the key for the memory size
KEY_HW_MEM_SIZE: Final[str] = "memSize"
#: the operating system in the sys-info section
SCOPE_OS: Final[str] = "os"
#: the operating system name
KEY_OS_NAME: Final[str] = "name"
#: the operating system version
KEY_OS_VERSION: Final[str] = "version"
#: the operating system release
KEY_OS_RELEASE: Final[str] = "release"
#: the python scope in the sys-info section
SCOPE_PYTHON: Final[str] = "python"
#: the python version
KEY_PYTHON_VERSION: Final[str] = "version"
#: the python implementation
KEY_PYTHON_IMPLEMENTATION: Final[str] = "implementation"

#: the resulting point in the solution space
SECTION_RESULT_Y: Final[str] = "RESULT_Y"
#: the resulting point in the search space
SECTION_RESULT_X: Final[str] = "RESULT_X"

#: all the section titles combined in one set
# noinspection PyDefaultArgument
# pylint: disable=[C3002]
_ALL_SECTIONS: Final[set[str]] = (lambda g=globals(): {  # type: ignore # noqa
    g[k] for k in g if k.startswith("SECTION_")})()  # type: ignore
