#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

double gamma_alpha;
double gamma_five_alpha;

double f(double, double, double);
double * fr_Euler_method(double, double, double, double);

int main(){
    double y0 = 0;
    double tfin = 1.0;
    double Delta_t = 1.0/640.0;
    int N = (int)ceil((tfin-Delta_t)/Delta_t);

    double alpha = 0.1;

    double * y = fr_Euler_method(y0, Delta_t, tfin, alpha);

    fstream myfile;
    myfile.open("datas.dat", ios::out);

    if (!myfile) {
        cout << "File not created!";
    }
    else {

        int k;
        for(k=0; k<N; k++)
            myfile << y[k] << "\n";

        myfile.close();
    }

    return 0;
}

double f(double t_, double y_, double alpha){
    return -y_ + (1/gamma_five_alpha)*pow(t_, 4 - alpha);
}

double * fr_Euler_method(double y0, double Delta_t, double tfin, double alpha){
    int N = (int)ceil((tfin-Delta_t)/Delta_t);

    /*  Memory allocates dynamically using malloc() */
    double *y = (double*) malloc(N * sizeof(double));
    double *t = (double*) malloc(N * sizeof(double));

    /* Initial conditions */
    y[0] = y0;
    t[0] = Delta_t;

    int k;

    #pragma omp critical
    for(k=1; k<N; k++)
        t[k] = t[k-1] + Delta_t;

    int n;
    int j;
    double bj = 0;
    double sum_bj = 0;

    gamma_alpha = tgamma(alpha + 1.0);
    gamma_five_alpha = tgamma(5.0 - alpha);

    #pragma omp master
    for(n=0; n<N; n++){
        sum_bj = 0;

        #pragma omp critical
        for(j=0; j<n+1; j++){
            bj = pow(n-j+1,alpha) - pow(n-j,alpha);
            sum_bj = sum_bj + bj*f(t[j], y[j], alpha);
            //sum_bj = sum_bj + bj*( -y[j] + (1/gamma_five_alpha)*pow(t[j],4-alpha) );
        }

        y[n+1] = y[0] + pow(Delta_t,alpha)*(1/gamma_alpha)*sum_bj;
    }

    /*y = NULL;
    t = NULL;

    free(y);
    free(t);*/

    return y;
}
