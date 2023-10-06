#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

void save_vectors(double *, double *, double *, double *, int);
void newton_polynomial(double *, double, double, double, double *);

int main(){
    /* Parameters Chen's model */
    double parameters[3] = {40, 3, 28};
    
    /* Parameters method */
    int tfin = 10;
    double h = 0.001;
    
    /* Initial conditions */
    double x0[3] = {-0.1, 0.5, -0.6};
    
    /* Fractional order */
    double alpha = 0.99;
    
    newton_polynomial(x0, tfin, h, alpha, parameters);
    
    return 0;
}

void save_vectors(double * t, double * y1, double * y2, double * y3, int size){
    FILE *fp;
    
    fp = fopen("datas.dat","w");
    
    for(int k=0; k<size; k++)
        fprintf(fp, "%f, %f, %f, %f\n",t[k], y1[k], y2[k], y3[k]);
    
    fclose(fp);
}

void newton_polynomial(double *x0, double tfin, double h, double alpha, double *parameters){
    int N = ceil((tfin-h)/h);
    
    /* Parameters */
    double A = parameters[0];
    double B = parameters[1];
    double C = parameters[2];
    
    /* Initial conditions */
    double *x = (double *)malloc(N*sizeof(double));
    x[0] = x0[0];
    
    double *y = (double *)malloc(N*sizeof(double));
    y[0] = x0[1];
    
    double *z = (double *)malloc(N*sizeof(double));
    z[0] = x0[2];
    
    double *t = (double *)malloc(N*sizeof(double));
    t[0] = 0;
    
    double gamma_alpha_1 = tgamma(alpha + 1.0);
    double gamma_alpha_2 = tgamma(alpha + 2.0);
    double two_gamma_alpha_3 = 2.0*tgamma(alpha + 3.0);
    
    /* Check if the memory has been successfully allocated by malloc or not */
    if (x == NULL || y == NULL || z == NULL){
        printf("Memory not allocated.\n");
        exit(0);
    }
    else{
    
        double start = omp_get_wtime();
        
        #pragma omp critical
        {
            for(int n=0; n<N-1; n++){
                double sum1a = 0; double sum2a = 0; double sum3a = 0;
                double sum1b = 0; double sum2b = 0; double sum3b = 0;
                double sum1c = 0; double sum2c = 0; double sum3c = 0;
            
                #pragma omp parallel for reduction(+ : sum1a,sum1b,sum1c,sum2a,sum2b,sum2c,sum3a,sum3b,sum3c)
                for(int j=0; j<=n; j++){
                    double f1 = A*(y[j] - x[j]);
                    double f2 = (C - A)*x[j] - x[j]*z[j] + C*y[j];
                    double f3 = x[j]*y[j] - B*z[j];
                    
                    double term1 = pow(n - j + 1,alpha) - pow(n - j,alpha);
                    double term2 = pow(n - j + 1,alpha)*(n - j + 3 + 2*alpha) - pow(n - j,alpha)*(n - j + 3 + 3*alpha);
                    double term3 = pow(n - j + 1,alpha)*(2*pow(n - j,2) + (3*alpha + 10)*(n - j) + 2*pow(alpha,2) + 9*alpha + 12) - 
                        pow(n - j,alpha)*(2*pow(n - j,2) + (5*alpha + 10)*(n - j) + 6*pow(alpha,2) + 18*alpha + 12);
                
                    sum1a += (f1*term1);
                    sum1b += ((f1 - f1)*term2);
                    sum1c += ((f1 - 2*f1 + f1)*term3);
                
                    sum2a += (f2*term1);
                    sum2b += ((f2 - f2)*term2);
                    sum2c += ((f2 - 2*f2 + f2)*term3);
                    
                    sum3a += (f3*term1);
                    sum3b += ((f3 - f3)*term2);
                    sum3c += ((f3 - 2*f3 + f3)*term3);
                } /* end for */
                
                t[n+1] = t[n] + h;
                x[n+1] = x[0] + pow(h,alpha)*((1/gamma_alpha_1)*sum1a + (1/gamma_alpha_2)*sum1b + (1/two_gamma_alpha_3)*sum1c);
                y[n+1] = y[0] + pow(h,alpha)*((1/gamma_alpha_1)*sum2a + (1/gamma_alpha_2)*sum2b + (1/two_gamma_alpha_3)*sum2c);
                z[n+1] = z[0] + pow(h,alpha)*((1/gamma_alpha_1)*sum3a + (1/gamma_alpha_2)*sum3b + (1/two_gamma_alpha_3)*sum3c);
            } /* end for */
            
        } /* end pragma omp critical */
        
        double end = omp_get_wtime();

        printf("It took %f\n", end-start);
        
    } /* end else */
    
    save_vectors(t,x,y,z,N);
    
    free(t);
    free(x);
    free(y);
    free(z);
}
 
