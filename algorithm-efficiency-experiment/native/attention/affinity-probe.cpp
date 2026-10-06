#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
// Separate diagnostic symbol; existing Metrics/Profile ABI and arithmetic unchanged.
extern "C" int refine_probe(int requested, int *rows) {
    int team=0;
    #pragma omp parallel num_threads(requested) shared(team,rows)
    {
        int i=omp_get_thread_num();
        #pragma omp single
        team=omp_get_num_threads();
        cpu_set_t mask; CPU_ZERO(&mask);sched_getaffinity(0,sizeof(mask),&mask);
        int bits=0;for(int c=0;c<4;c++)if(CPU_ISSET(c,&mask))bits|=1<<c;
        rows[i*4]=static_cast<int>(syscall(SYS_gettid));rows[i*4+1]=sched_getcpu();
        rows[i*4+2]=bits;rows[i*4+3]=omp_get_place_num();
    }
    return team;
}
