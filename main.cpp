#include <iostream>
#include <armadillo>
#include <omp.h>
#include "readFiles.h"

#define ROWS 512
#define COLS 512
#define N 512

using namespace std;
using namespace arma;

void Dx(cx_mat & d, cx_mat & U);
void Dxt(cx_mat & d, cx_mat & U);
void Dy(cx_mat & d, cx_mat & U);
void Dyt(cx_mat & d, cx_mat & U);
void shrink2(cx_mat & S, cx_mat & SS, cx_mat & x, cx_mat & y, cx_mat & bdx, cx_mat & bdy, double lambda);

const double sparsity = 0.25f;
const double mu = 0.1f;
const double lambda = 0.1f;
const double gammaBregman = 0.0001f;
const int nBreg = 5;
const int nInner = 25;

int main(int argc, char** argv){

  mat image(ROWS,COLS, fill::zeros);

  double * matData = (double *) malloc(COLS*ROWS*sizeof(double));
  const unsigned int totalByteSize = COLS*ROWS;
  char * filename = "D.bin";
  readBinaryMatrix(matData, filename, totalByteSize, sizeof(double));
  cout << "po wczytaniu" << endl;

  for(int i=0; i<ROWS; i++){
      for(int j=0; j<COLS; j++){
          image(i,j) = matData[i*ROWS +j];
      }
  }

  free(matData);

  //ustawienie image
  /*
  unsigned int start = COLS/4;
  unsigned int end = 3*ROWS/4;
  for(int i=start-1;i < end; i++){
    for(int j=start-1; j<end; j++)
        image(i,j) = 255.0;
  }
  */
  //image.save("image.mat", raw_binary);


  mat R = randu<mat>(ROWS,COLS);
  for(int i=0; i<ROWS; i++){

      for(int j=0; j<COLS; j++){
          if(R(i,j)<sparsity)
              R(i,j) = 1.0;
          else
              R(i,j) = 0.0;
      }
  }

  //R.save("R.mat", raw_binary);

  cx_mat imageFFT = fft2(image);
  imageFFT = imageFFT % R;
  imageFFT = imageFFT / N;

  double start_t = omp_get_wtime();

  //aglorytm
  cx_mat F0 = imageFFT;
  cx_mat U(ROWS,COLS, fill::zeros);
  cx_mat X(ROWS,COLS, fill::zeros);
  cx_mat Y(ROWS,COLS, fill::zeros);
  cx_mat BX(ROWS,COLS, fill::zeros);
  cx_mat BY(ROWS,COLS, fill::zeros);

  // Build Kernels

  double scale = sqrt(ROWS*COLS);
  //murf = ifft2(mu*(conj(R).*f))*scale;
  cx_mat MURF = ifft2(mu*(R%imageFFT))*scale;


  cx_mat UKER(ROWS,COLS, fill::zeros);
  UKER(0,0) = 4.0; UKER(0,1) = -1.0; UKER(1,0) = -1.0; UKER(ROWS-1,0) = -1.0; UKER(0,COLS-1) = -1.0;
  //uker = mu*(conj(R).*R)+lambda*fft2(uker)+gamma;
  UKER = fft2(UKER)*lambda + gammaBregman + mu*(R % R);

  //helper matrixes
  cx_mat Xu, Yu, RHS;

  //  Do the reconstruction
      for (int outer = 0; outer < nBreg; outer++){
          for (int inner = 0; inner < nInner; inner++){
              // update u
              Xu = X - BX;
              Dxt(X, Xu);

              Yu = Y - BY;
              Dyt(Y, Yu);
              RHS = MURF + lambda*(X+Y) + gammaBregman*U;
              U = ifft2(fft2(RHS)/UKER);

              // update x and y
              //dx = Dx(u); u nas dx jest Xu
              Dx(Xu, U);
              BX = BX + Xu;
              //dy = Dy(u); podobnie dy jest Yu
              Dy(Yu, U);
              BY = BY + Yu;
              //[x,y] = shrink2( dx+bx, dy+by,1/lambda);
              shrink2(Xu, Yu, X, Y, BX, BY, 1.0/lambda);

              // update bregman parameters
              BX = BX - X;
              BY = BY - Y;
          }

          //f = f+f0-R.*fft2(u)/scale;
          imageFFT = imageFFT + F0 - (R % (fft2(U))) / scale;
          //murf = ifft2(mu*R.*f)*scale;
          MURF = ifft2( mu*(R % imageFFT)) * scale;
  }
  // time measure

  double end_t = omp_get_wtime();
  cout << "total time: " << end_t - start_t << endl;
  //------------
  mat realU = real(U);
  mat imagU = imag(U);
  realU.save("Ureal.mat", raw_binary);
  imagU.save("Uimag.mat", raw_binary);

  return 0;
}

void Dx(cx_mat & d, cx_mat & U){
    d.cols(1, COLS-1) = U.cols(1, COLS-1) - U.cols(0, COLS-2);
    d.col(0) = U.col(0) - U.col(COLS-1);
}

void Dxt(cx_mat & d, cx_mat & U){

    d.cols(0, COLS-2) = U.cols(0, COLS-2) - U.cols(1,COLS-1);
    d.col(COLS-1) = U.col(COLS-1) - U.col(0);
}


void Dy(cx_mat & d, cx_mat & U){

    d.rows(1,ROWS-1) = U.rows(1,ROWS-1) - U.rows(0,ROWS-2);
    d.row(0) = U.row(0) - U.row(ROWS-1);
}


void Dyt(cx_mat & d, cx_mat & U){

    d.rows(0,ROWS-2) = U.rows(0,ROWS-2) - U.rows(1,ROWS-1);
    d.row(ROWS-1) = U.row(ROWS-1) - U.row(0);
}

void shrink2(cx_mat & S, cx_mat & SS, cx_mat & X, cx_mat & Y, cx_mat & BDX, cx_mat & BDY, double lambda){

    S = sqrt(X%conj(X) + Y%conj(Y));
    SS = S-lambda;
    SS = SS % conv_to<mat>::from(real(SS) > 0.0);

    S = S + conv_to<mat>::from(real(S) < lambda);
    SS = SS / S;

    X = SS % BDX;
    Y = SS % BDY;
}

