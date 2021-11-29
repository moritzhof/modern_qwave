
#include "ddgemm.hpp"

#include <cassert>

namespace qwv{

//note: I kept the interface the same for now, parameters that are not used are marked with an underscore in front
// (for instance I don't do any timing here anymore, and we assume sequential execution)
void parpack_interface(int n, OperatorBase<double> *A_op, int k,long int calcEigVec, int maxit, double sigma,int p,int randVec, int *iparam, double *dr, double *di,double *eigVecReal, double *eigVecIm, double LEigValReal, double *_tsub,double *_tcomm,int my_rank, int pr,int _ncores){
    //ARPACK interface to calculate eigenvalues of a real, non-symmetric matrix
    /* n: Size of the Matrix
     * k: Number of requested eigenvalues
     * maxit: maximum number of iterations
     * sigma: Eigenvalues close to sigma are calculated -> not implemented yet
     */

        const double  EPS=DBL_EPSILON;
        int mode,ido,nev,ncv,lworkl,ldv,info,eigs_display,i;
        double tol,sigmar,sigmai;
        int32_t rvec, *select;
        int *nstartInd;
        double *v, *workd, *workl, *resid,*workev,*temp;

        int ipntr[14]={0};
        mode=1;                        //calculate largest eigenvalue
        ido=0;                         //reverse communication parameter, initial value
    //    p=fmin(fmax(2*k+1,20),n);        //number of Lanczos vectors
    //    maxit=fmax(300,ceil(2*n/fmax(p,1)));
        const char* bmat="I";                    //standard eigenvalue problem
        const char* which="LM";                    //eigenvalues with largest magnitude
        const char* howmny="A";                    //Compute NEV Ritz vectors (no Schur vectors)
        nev=k;                         //number of requested eigenvalues
        ncv=p;                        //number of Lanczos vectors
        iparam[0]=1;            //ishift=1, no handling of ido=3 necessary
        iparam[2]=maxit;        //maximal number of iterations
        iparam[6]=mode;
        rvec=calcEigVec;                    //only Ritz values, no Ritz/Schur vectors are calculated
        lworkl=3*p*(p+2);
        sigmar=sigma;
        sigmai=0;

        int nloc = (int)A_op->num_local_rows();
        // we don't support parallelism here right now,
        // but it would be easy to add because we already
        // call parpack instead of arpack.
        assert(nloc==n);

        //Initialize PNAUPD paramareters
        ldv=nloc;
        select=(int32_t*)calloc(p,sizeof(int));
        v=(double*)calloc(nloc*ncv,sizeof(double));
        workd=(double*)calloc(3*nloc,sizeof(double));
        workl=(double*)calloc(lworkl,sizeof(double));
        resid=(double*)calloc(nloc,sizeof(double));
        workev=(double*)calloc(3*ncv,sizeof(double));

        printf("Calloc resources\n");

        tol=EPS;
        info=0; //0 if initial vector resid is not initialised
        if(randVec==1){
            info=1;
            generateRandomVector4D(n,nloc,0,resid);
        }
        eigs_display=2;


        long int len_bmat=strlen(bmat);
        long int len_which=strlen(which);
        long int len_howmny=strlen(howmny);

        //Convert MPI communicator from C handle to Fortran integer
        MPI_Fint comm;
        comm=MPI_Comm_c2f(MPI_COMM_WORLD);

//former par_parpack_interface
{
    FILE *fpStat;
    const char* fpStatName="Status.txt";
    if(my_rank==0){
        fpStat=fopen(fpStatName,"a+");
        fprintf(fpStat,"------------------ STATUS ---------------------\n\n");
        fclose(fpStat);
        }


    int eigs_iter=0;
    while(ido!=99){
        pdnaupd_(&comm,&ido,(char*)bmat,&nloc,(char*)which,&nev,&tol,resid,&ncv,v,&ldv,iparam,ipntr,workd,workl,&lworkl,&info,len_bmat,len_which);
        if (info<0){
            //ERROR: dnaupd
            if (my_rank==0){
                printf("ARPACK dnaupd routine error for info= %i\n",info);
                }
            break;
            }
            // right-hand side x, left-hand side y for operation y=A*x
            double* rhs = workd+ipntr[0]-1;
            double* lhs = workd+ipntr[1]-1;
            switch (ido){
                case -1:
                  // same as 1
                case 1:
                    A_op->apply(1.0,rhs, 0.0, lhs);
                    break;
                case 99:
                    break;
                    //ARPACK has converged
                default:
                    //ERROR unknown ido
                    printf("Unknown ido after calling dnaupd routine\n");
                    break;
                }

            eigs_iter++;
            if (eigs_display==0 && my_rank==0){
                displayRitzValues(ipntr,eigs_iter,ido,ncv,nev,workl);
                }
            if (eigs_display==1 && my_rank==0){
                printf("Iteration %i for the Ritz values of the %i-by-%i matrix:\n",eigs_iter,ncv,ncv);
                }
            if(eigs_iter%1000==0 && my_rank==0){
                fpStat=fopen(fpStatName,"a+");
                fprintf(fpStat,"Iteration %i: a few Ritz values of the %i-by-%i matrix:\n",eigs_iter,ncv,ncv);
                for(int j=ncv-1;j>fmax(ncv-nev-1,ncv-30-1);j--){
                    fprintf(fpStat,"%.12f %.5f\n",workl[ipntr[5]+j-1]+LEigValReal,workl[ipntr[6]+j-1]);
                    }
                fprintf(fpStat,"%.12f %.5f\n",workl[ipntr[5]+ncv-nev-1]+LEigValReal,workl[ipntr[6]+ncv-nev-1]);
                fprintf(fpStat,"\n");
                fclose(fpStat);
                }
        }// end while

    if(my_rank==0){
        fpStat=fopen(fpStatName,"a+");
        fprintf(fpStat,"Total number of converged Ritz values: %i\n",iparam[4]);
        fclose(fpStat);
        if(info!=0){
            printf("Error durring pndaupd routine, info=%i\n",info);
            printf("Number of Arnoldi iterations taken: %i\n",iparam[2]);
            }
        }
    pdneupd_(&comm,&rvec,(char*)howmny,select,dr,di,v,&ldv,&sigmar,&sigmai,workev,(char*)bmat,&nloc,(char*)which,
        &nev,&tol,resid,&ncv,v,&ldv,iparam,ipntr,workd,workl,&lworkl,&info,len_howmny,len_bmat,len_which);
    if(info!=0){
        printf("Error during pdneupd routine\n");
        }
} // end former par_parpack_intrface

        free(select);free(v);free(workl);free(resid);free(workev);
        return;
    }
}
