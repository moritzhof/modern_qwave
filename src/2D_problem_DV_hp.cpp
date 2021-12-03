#include <iostream>

#include "eigensolver_interface.hpp"
#include "TensorOperator.hpp"
#include "buildOperator.hpp"
#include "buildPreconditioner.hpp"
#include "mBuildMatrix.hpp"
#include "mMatrixOperation.hpp"
#ifdef USE_PHIST
#include "phist_kernels.h"
#endif

using namespace QWaves;

int main(int argc, char* argv[]){

std::cout << "#####################  Beginning Program ####################\n";

//############################ read in arguments/parameters #################################

   if (argc<14){
      fprintf(stderr, "USAGE %s <nR> <nr> <NumOfEigval> <NumOfLanczosVectors> <pot> <Eb> <V12> <V13> <V23> <q> <alpha> <parity1> <parity2>", argv[0]);
      return 1;
    }

        int nR=atoi(argv[1]);
        int nr=atoi(argv[2]);

//################################## MPI & PHIST #############################################

        int rank;
        int np;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int iflag=0;
        PHIST_ICHK_IERR(phist_kernels_init(&argc,&argv,&iflag),iflag);

       MPI_Comm_size(MPI_COMM_WORLD, &np);

//############################################################################################
       int i;
       int NumOfEigval, NumOfLanczosVectors,StoreEigVec;
       long int CalcEigVec;
       double *EigValReal;
       double *EigValIm;
       double *EigVecReal,*EigVecIm;

       NumOfEigval=atoi(argv[3]);
       NumOfLanczosVectors=atoi(argv[4]);
       CalcEigVec=0;
       StoreEigVec=0;
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

//##############################################################################################

       double q=atof(argv[10]);
       double alpha=1.0/atof(argv[11]); //alpha=m/M<<1
       double LR=6*0.5*sqrt(1/(2*Eb))*(2+alpha)/sqrt(1+alpha);
       double Lr=6*0.25*sqrt(1/(2*Eb))*(2+alpha)*1/sqrt(1+alpha);

       double a1=-alpha/(1+alpha);
       double a2=-0.25*(2+alpha)/(1+alpha);

       int parity1=atoi(argv[12]);
       int parity2=atoi(argv[13]);

//############################################################################################

// scope to make sure everything is deleted before phist is finalized
{

// create the parallel distribution and iterator object
TensorMap4D map(std::tie(nr,nr),std::tie(nR,nR),MPI_COMM_WORLD);

std::cout << "#################### Building Potential ##################### \n";

std::vector<double> V4D = buildGaussianPotential4D<double>(nR,nr,LR,Lr,V12,V13,V23, map);


std::cout << "#################### Building Operator ###################### \n";

std::unique_ptr<Kron4D<double>> H4_op = buildOperator4D(nR, nr, LR, Lr, parity1, parity2, a1, a2, V4D);

std::cout << "################# Construction Complete ##################### \n";

//################## if preconditioner ##################################
// double sigma = 0.0;
// std::unique_ptr<OperatorBase<double>> H2_prec = buildShiftedSylvesterPrecond2D(*H2_op, 0.0);
//########################################################################

phist_interface(H4_op.get(), nullptr, "2DjadaOpts-tbbs.txt", NumOfEigval, CalcEigVec,
        EigValReal, EigValIm, EigVecReal, EigVecIm,
        H4_op->map().get_phist_map());
}
PHIST_ICHK_IERR(phist_kernels_finalize(&iflag),iflag);

std::cout <<std::endl<<std::endl;
std::cout << "-------------------------\n";
std::cout << "Model parameters used: "<<std::endl;
std::cout << "-------------------------\n";
std::cout << "Eb="<<Eb<<std::endl;
std::cout << "alpha="<<alpha<<std::endl;
std::cout << "LR="<<LR<<std::endl;
std::cout << "Lr="<<Lr<<std::endl;
std::cout << "V12="<<V12<<std::endl;
std::cout << "V13="<<V13<<std::endl;
std::cout << "V23="<<V23<<std::endl;
std::cout << "a1="<<a1<<std::endl;
std::cout << "a2="<<a2<<std::endl;
std::cout << "-------------------------\n";


std::cout << "PHIST found "<<NumOfEigval<<" eigenpairs"<<std::endl;
std::cout << "to the desired tolerance."<<std::endl;
std::cout << std::endl;
std::cout << "=========================\n";
std::cout << "Scaled eigenvalues E/Eb:\n"<<std::endl;
std::cout << "-------------------------\n";
for (int i=0; i<NumOfEigval;i++)
{
  fprintf(stdout,"%d\t%25.16e\n",i,EigValReal[i]/Eb);
}
std::cout << "=========================\n";

free(EigValReal);
free(EigValIm);
free(EigVecReal);
free(EigVecIm);
MPI_Finalize();
return 0;
}
