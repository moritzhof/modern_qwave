
//////////////////////////////////////////////////
//                                              //
//                                              //
//           THREE-BODY BOUND STATES            //
//                                              //
//                                              //
//           ____              ____             //
//          /    \            /    \            //
//         /      \_ _ _ _ _ /      \           //
//         \      /          \      /           //
//          \____/ \        / \____/            //
//                  \      /                    //
//                   \ __ /                     //
//                    /  \                      //
//                    \__/                      //
//                                              //
//                                              //
//                                              //
//////////////////////////////////////////////////
// Author: Matthias Zimmermann                  //
// Ulm, January 2016                            //
//                                              //
//////////////////////////////////////////////////

#include "eigensolver_interface.hpp"
#include "SparseMatrixWrapper.hpp"
#include "TensorOperator.hpp"
#include "buildOperator.hpp"
#include "buildPreconditioner.hpp"
#include "mBuildMatrix.hpp"
#include "mMatrixOperation.hpp"
#ifdef USE_PHIST
#include "phist_kernels.h"
#endif


int main(int argc, char *argv[]){

    if (argc<13)
    {
      fprintf(stderr, "USAGE %s <nR> <nr> <NumOfEigval> <NumOfLanczosVectors> <pot> <Eb> <V12> <V13> <V23> <q> <alpha> <parity1> <parity2>",argv[0]);
      return 1;
    }

        int nR=atoi(argv[1]);
        int nr=atoi(argv[2]);

/*%%%%%%%%%% INITIALIZE MPI %%%%%%%%%%%%%%%%%%*/
       int  my_rank; /* rank of process */
       int  np;       /* number of processes */
       int  ncores=64;  /* number of threads per node */
       /* start up MPI */

       MPI_Init(&argc, &argv);

       /* find out process rank */
       MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

#if USE_PHIST
        /* depending on the backend (kernel library), phist may require some
           initialization (e.g. timing, I/O and computing resource management)
         */
        int iflag=0;
        PHIST_ICHK_IERR(phist_kernels_init(&argc,&argv,&iflag),iflag);;
#endif
       /* find out number of processes */
       MPI_Comm_size(MPI_COMM_WORLD, &np);

/*%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%*/

       /*%%%%%%%%%%% Global DECLARATION %%%%%%%%%*/
       int i;
       int NumOfEigval, NumOfLanczosVectors,StoreEigVec;
       long int CalcEigVec;
       double *EigValReal;
       double *EigValIm;
       double *EigVecReal,*EigVecIm;
       NumOfEigval=atoi(argv[3]);
       NumOfLanczosVectors=atoi(argv[4]);
       CalcEigVec=1;
       StoreEigVec=1;
       EigValReal=(double*)calloc(NumOfEigval+1,sizeof(double));
       EigValIm=(double*)calloc(NumOfEigval+1,sizeof(double));
       EigVecReal=(double*)calloc((NumOfEigval+1)*nR*nr,sizeof(double));
       EigVecIm=(double*)calloc((NumOfEigval+1)*nR*nr,sizeof(double));

       double LEigValReal;
       double LEigValIm;
       LEigValReal=0;
       LEigValIm=0;
       struct timespec t[10];
       double tsub1[4];
       double tsub2[4];
       double tcomm1[10]={0};
       double tcomm2[10]={0};
       char   pot=argv[5][0]; //(G=Gaussian, H=Harmonic, L=Lorentzian)
       double Eb=pow(10,atof(argv[6]));
       double V12=atof(argv[7]);
       double V13=atof(argv[8]);
       double V23=atof(argv[9]);

       SparseMatrix *Vsp2D;
       std::vector<double> V2D;
       double q=atof(argv[10]);
double alpha=1/atof(argv[11]); //alpha=m/M<<1
double LR=6*0.5*sqrt(1/(2*Eb))*(2+alpha)/sqrt(1+alpha);
double Lr=6*0.25*sqrt(1/(2*Eb))*(2+alpha)*1/sqrt(1+alpha);
//       double LR=2;
//       double Lr=2;

double a1=-alpha/(1+alpha);
double a2=-0.25*(2+alpha)/(1+alpha);

int parity1=atoi(argv[12]);
int parity2=atoi(argv[13]);
/*%%%%%%%%%% STORE INFORMATION %%%%%%%%%%%%%%%%%*/
char fpEig[150],fpEigVecName[150],fpEigValName[150],fpInfName[150],fpTimeName[150],fpWavePropName[150];
sprintf(fpEig,"2D_parity_%c_V12_%.4f_V13_%.4f_V23_%.4f_alpha_%.4f_nR_%i_nr_%i_kE_%i_LR_%.3f_Lr_%.3f_p1_%i_p2_%i",pot,V12,V13,V23,alpha,nR,nr,NumOfEigval,LR,Lr,parity1,parity2);
sprintf(fpEigVecName,"%s_EigenVectors.csv",fpEig);
sprintf(fpEigValName,"%s_EigenValues.csv",fpEig);
sprintf(fpInfName,"%s_Info.txt",fpEig);
sprintf(fpTimeName,"%s_Timing.txt",fpEig);
sprintf(fpWavePropName,"%s_Wave_Function_Properties.csv",fpEig);
if(my_rank==0){
      store_information_1(np,ncores,parity1,parity2,alpha,pot,q,V12,V13,V23,nR,nr,LR,Lr,NumOfEigval,NumOfLanczosVectors,fpInfName);
}
/*%%%%%%%%%% BUILD DIFFERENTIAL MATRIX %%%%%%%%%%%%%%%%%*/
clock_gettime(CLOCK_MONOTONIC, &t[0]);

//std::cout << "###########################################################\n";

//int N = nR*nR;
//int n = nr*nr; 
//std::cout << " 2D PROBLEM: SEGFAULT ???? \n";
//matrix<double> Gauss3B2D(64*64, 32*32); 
//Gauss3B2D = buildGaussianPotential3B_2D<double>(64, 32,LR, Lr, V12, V13, V23);
//std::cout << "#############################################################\n";
switch(pot){
      case 'G':
	     Vsp2D=buildGaussianPotential2D(nR,nr,LR,Lr,V12,V13,V23);
	     break;
      case 'H':
	     a1=-0.5;
	     a2=-0.5;
	     Vsp2D=buildHarmonicPotential2D(nR,nr,LR,Lr,1.0,1.0);
	     break;
      case 'L':
	     Vsp2D=buildLorentzianPotential2D(nR,nr,LR,Lr,V12,V13,V23,q);
	     break;
      default:
	     return 0;
	     break;
}

std::cout << "BUILD POTENTIAL\n";

clock_gettime(CLOCK_MONOTONIC, &t[1]);

// for the operator class we need the diagonal of the potential operator
//  as a std::vector. So we convert it and complain if it's not a diagonal matrix
// for the moment.
V2D.resize(Vsp2D->dim);
        if (Vsp2D->nz!=1) throw std::runtime_error("potential matrix must have one entry per row at the moment.");
        bool is_diagonal=true;
        for (int i=0; i<Vsp2D->dim; i++)
        {
          if (Vsp2D->columnInd[i]!=i)
          {
            is_diagonal=false;
          }
          V2D[i]=Vsp2D->value[i];
        }
        if (!is_diagonal) throw std::runtime_error("potential matrix must have non-zero entries only on the diagonal.");

       std::unique_ptr<QWaves::Kron2D<double>>
       H2_op=QWaves::buildOperator2D(nR,nr,LR,Lr,parity1,parity2,a1,a2,V2D);
       DeleteSparseMatrix(Vsp2D);
       MPI_Barrier(MPI_COMM_WORLD);
       clock_gettime(CLOCK_MONOTONIC, &t[2]);
       

       //this is a place holder, jonas. you forgot to add it
     //  double simga = 0.131234; 
     //  std::unique_ptr<QWaves::OperatorBase<double>> H2_prec = buildShiftedSylvesterPrecond2D(*H2_op, simga);



       QWaves::phist_interface(H2_op.get(), nullptr, "jadaOpts-tbbs.txt",NumOfEigval,CalcEigVec,EigValReal,EigValIm,EigVecReal,EigVecIm);

       clock_gettime(CLOCK_MONOTONIC, &t[5]);

       if (my_rank==0) printf("%d eigenvalues computed\n",NumOfEigval);
       for (int i=0; i<NumOfEigval; i++){
         fprintf(stdout,"%d\t%16.8e\n",i,EigValReal[i]);
       }
//        MPI_Barrier(MPI_COMM_WORLD);

/*%%%%%%% STORE RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
       if(my_rank==0){
              printf("Node %i: Store Information 2\n",my_rank);
              //store_information_2(np,nR,nr,iparam1,iparam2,LEigValReal,tsub1,tsub2,fpInfName);
              printf("Node %i: Store Eigenvalues\n",my_rank);
              store_eigenvalues_2D(nR,nr,NumOfEigval,nullptr,EigVecReal,EigValReal,EigValIm,fpEigValName);
              if(CalcEigVec==1 && StoreEigVec==1){
                     store_eigenvectors_2D_parity_opt(nR,nr,NumOfEigval,LR,Lr,parity1,parity2,EigVecReal,fpEigVecName);
              }
              if(CalcEigVec==1){
//              store_wave_function_properties(N,NumOfEigval,L,EigVecReal,fpWavePropName);
              }
              clock_gettime(CLOCK_MONOTONIC, &t[6]);
              store_timing(t,tsub1,tsub2,tcomm1,tcomm2,fpTimeName);
       }
/*%%%%%%% CLEAN UP %%%%%%%%%%%%%%%%%%%%*/
       free(EigValReal);free(EigValIm);
       free(EigVecReal);free(EigVecIm);

#if USE_PHIST
        PHIST_ICHK_IERR(phist_kernels_finalize(&iflag),iflag);
#endif

       /* shut down MPI */
       MPI_Finalize();

       return 0;
}
