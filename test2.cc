#include <string>
#include <cblas.h>
#include <lapacke.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>
#include <string.h>
#include <chrono>

//#include "lapack_aux.h"
 

lapack_int printMatrix(int matrix_layout, lapack_int m, lapack_int n, double* a_matrix, lapack_int lda){
 
        if(matrix_layout == LAPACK_COL_MAJOR){
                for(int i=0;i<m;i++){
                        printf("\n");
                        for(int j=0;j<n; j++){
                                printf("  %8.8f  ",a_matrix[i + j*lda]);
                        }
                }
        }else if(matrix_layout == LAPACK_ROW_MAJOR){
                for(int i=0;i<m;i++){
                        printf("\n");
                        for(int j=0;j<n; j++){
                                printf("  %8.8f  ",a_matrix[i*lda + j]);
                        }
                }
 
        }else{
                printf("Parameter matrix_layout is error!\n");
 
                return -1;
        }

        printf("\n");
 
        return 0;
}


void transpose_mat(const double *A, const int nrows, const int ncols, double *B, int nthreads)
{
    if (nrows >= ncols)
    {
        //#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int row = 0; row < nrows; row++)
            cblas_dcopy(ncols, A + (size_t)row*(size_t)ncols, 1, B + row, nrows);
    }
    
    else
    {
        //#pragma omp parallel for schedule(static) num_threads(nthreads
        for (int col = 0; col < ncols; col++)
            cblas_dcopy(nrows, A + col, ncols, B + (size_t)col*(size_t)nrows, 1);
    }
}



int getAffineTransform(double *src,double *dst,double *M)
{
    //
    double C[6] = {0.00};
    int ipiv[3]={0};
    int status = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,3,3,src,3,ipiv);
    LAPACKE_dgetri(LAPACK_ROW_MAJOR,3,src,3,ipiv);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  
                3, 2, 3, 1.0, (double *)src, 3, (double *)dst, 2, 0.0, (double *)C, 2);
    //double C_T[9] = {0.00};
    //cblas_comatcopy(CblasRowMajor,CblasTrans,2,3,1.0,C,3,M,3); can`t using instead of mkl

    transpose_mat(C, 3, 2, M, 16);
    
    return 1;
    //
}

template<typename T>
T * choose_funtion(T * matrix_a,int dim_s,int x,int y,int out_cols)
 {
     int out_row=x;
     int out_col=y;
     int dim =3;
    return matrix_a+out_cols*out_col*dim+out_row*dim+dim_s;
}

int * warpAffine(double * image, int image_rows,int image_cols,double *M, int M_rows,int M_cols,int out_rows, int out_cols)
{   
     int dim=3;

    //rows, cols, *_ = image.shape

     int rows = image_rows;
     int cols = image_cols;

    //out_rows, out_cols, *_ = output_shape
    //print("rows cols",rows,cols)
    //print("out_rows out_cols",out_rows,out_cols)

    //output = np.zeros(output_shape, dtype=image.dtype)

    //double * output=new double [out_rows*out_cols*3];

    int * output = new int [out_rows*out_cols*3];
    memset(output,0,out_rows*out_cols*3);

    int index=0;
    for(int out_row=0;out_row<out_rows;out_row++)
     {
         for(int out_col=0;out_col<out_cols;out_col++)
         {
            
             //in_col, in_row, _ = np.dot(M, [out_col, out_row, 1]).astype(int)

             double temp_m[3] = {out_col,out_row,1};//need trans 

             double C[3]={0};

             cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  
                 3, 1, 3, 1.0, (double *)M, 3, (double *)temp_m, 1, 0.0, (double *)C, 1);
            
             int in_col = C[0];
             int in_row = C[1];

             

             if (in_row >=0 && in_row<rows && in_col >=0 && in_col< cols)
             {   
                 //r
                 //double *r_output=matrix_a+out_cols*out_row*dim+out_col*dim+0;
                
                 //g
                 //double *g_output=matrix_a+out_cols*out_row*dim+out_col*dim+1;
                 //b 
                 //double *b_output=matrix_a+out_cols*out_row*dim+out_col*dim+2;

                if(index<20)
                {
                    //printf(" out_row:%d out_col:%d in_col:%d in_row:%d\n",out_row,out_col,in_col,in_row);
                    printf("image_r:%f image_g:%f image_b:%f \n",*(choose_funtion(image,0,in_col,in_row,image_cols)),*(choose_funtion(image,1,in_row,in_col,image_cols)),*(choose_funtion(image,2,in_col,in_row,image_cols)));
                }
                index++;

                 *(choose_funtion(output,0,out_col,out_row,out_cols))=*(choose_funtion(image,0,in_col,in_row,image_cols));
                 *(choose_funtion(output,1,out_col,out_row,out_cols))=*(choose_funtion(image,1,in_col,in_row,image_cols));
                 *(choose_funtion(output,2,out_col,out_row,out_cols))=*(choose_funtion(image,2,in_col,in_row,image_cols));
             }
         }
     }
     return output;
 }

template<typename T>
void save_data_to_local(T * data,int size,std::string name)
{

FILE *file = fopen((std::string("data")+name+std::string(".yuv")).c_str(), "wb");



size_t written = fwrite(data, sizeof(T), size, file);


if (written != size) {


    perror("Failed to write data");


}

fclose(file);

}



unsigned char * mat2rgbsave(cv::Mat &image)
{
    int h=image.rows;
    int w=image.cols;
    unsigned char * data=new unsigned char[h*w*3];
    for(int row=0;row<h;row++)
    {
        for(int col=0;col<w;col++)
        {
            *(choose_funtion(data,0,col,row,w))=*(choose_funtion((unsigned char *)image.data,0,col,row,w));
            *(choose_funtion(data,1,col,row,w))=*(choose_funtion((unsigned char *)image.data,1,col,row,w));
            *(choose_funtion(data,2,col,row,w))=*(choose_funtion((unsigned char *)image.data,2,col,row,w));
        }
    }

    save_data_to_local(data,h*w*3,"origin");
    return data;
}


double * mat2rgbdouble(cv::Mat &image)
{   
    int index=0;
    int h=image.rows;
    int w=image.cols;
    double * data=new double[h*w*3];
    for(int row=0;row<h;row++)
    {
        for(int col=0;col<w;col++)
        {

            
            *(choose_funtion(data,0,col,row,w))=(int)*(choose_funtion((unsigned char *)image.data,0,col,row,w));
            *(choose_funtion(data,1,col,row,w))=(int)*(choose_funtion((unsigned char *)image.data,1,col,row,w));
            *(choose_funtion(data,2,col,row,w))=(int)*(choose_funtion((unsigned char *)image.data,2,col,row,w));
            if(index<20)
            {
                //printf(" out_row:%d out_col:%d in_col:%d in_row:%d\n",out_row,out_col,in_col,in_row);
                //printf("image_r:%d image_g:%d image_b:%d \n",(int)*(choose_funtion((unsigned char *)image.data,0,row,col,w)),(int)*(choose_funtion((unsigned char *)image.data,1,row,col,w)),(int)*(choose_funtion((unsigned char *)image.data,2,row,col,w)));
                //printf("image_r:%f image_g:%f image_b:%f \n",*(choose_funtion(data,0,row,col,w)),*(choose_funtion(data,1,row,col,w)),*(choose_funtion(data,2,row,col,w)));
            }
            index++;
        }
    }

    
    return data;


}


void double_to_int(double * data_origin,int size_,int * data_int)
{   
    printf("con 1\n");
    for(int num_data=0;num_data<size_;num_data++)
    {
        *(data_int+num_data)=(int)(*(data_origin+num_data));
        if(num_data<20)
        {
            printf("%f\n",*(data_origin+num_data));
        }
    }
    printf("con 2\n");
}


void int_to_unsignedchar(int * intarray,unsigned char * array,int size)
{
    for(int time=0;time<size;time++)
    {
        *(array+size)=(unsigned char)intarray[size];
    }
}


int main(int argc,char** argv)
{
	setlocale(LC_ALL,"");
	double a[] = 
	{
		0,0,1,
		3,3,1,
		0,3,1
	};
	int m = 3;
	int n = 3;
	int lda = 3;
	int ipiv[3];
	int info;
	printMatrix(LAPACK_ROW_MAJOR,3,3,a,3);
    printf("===========\n");
	info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,m,n,a,lda,ipiv);
	printMatrix(LAPACK_ROW_MAJOR,3,3,a,3);
    printf("===========\n");
	info = LAPACKE_dgetri(LAPACK_ROW_MAJOR,m,a,lda,ipiv);
	printMatrix(LAPACK_ROW_MAJOR,3,3,a,3);
    printf("===========\n");

    double src[9]={0.0,0.0,1.0,
            3.0, 0.0, 1.0,
            0.0, 3.0, 1.0};
    
    double dst[6] ={3.0,1.0,
                3.0,3.0,
                0.0,0.0};
    
    double M[6] = {0};
    std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
    int status=getAffineTransform(src,dst,M);
    std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();
    std::printf("get affinetime = %ld ms\n", std::chrono::duration_cast</*std::chrono::microseconds*/std::chrono::milliseconds>(finishTP1 - startTP).count());
                    
    printf("the status:d\n",status);

    printMatrix(LAPACK_ROW_MAJOR,2,3,M,3);
    
    printf("run w\n");
    cv::Mat im = cv::imread("panda.jpg");
    unsigned char * data_image=mat2rgbsave(im);
    printf("run 0\n");
    double *im_buffer=mat2rgbdouble(im);
    printf("run 1\n");

    double M_t[] = {0.70710677,0.70710677,-400.0,
                    -0.70710677,0.70710677,250.0,
                    0,0,1};

                        double M_t2[] = {0.0,1.0,/*-400.0*/0,
                    -1.0,0,/*250.0*/0,
                    0,0,1};

    //(667, 825, 3)

    int index_=0;
    std::chrono::time_point<std::chrono::system_clock> startTP2 = std::chrono::system_clock::now();
    int * after_affine=warpAffine(im_buffer, im.rows,im.cols,M_t, 3,3,825,667);
    std::chrono::time_point<std::chrono::system_clock> finishTP2 = std::chrono::system_clock::now();
    std::printf("warpaffine time = %ld ms\n", std::chrono::duration_cast</*std::chrono::microseconds*/std::chrono::milliseconds>(finishTP2 - startTP2).count());
    // for(int out_row=0;out_row<825;out_row++)
    // {
    //      for(int out_col=0;out_col<667;out_col++)
    //      {
    //         if(index_<20)
    //         {
    //             //printf(" out_row:%d out_col:%d in_col:%d in_row:%d\n",out_row,out_col,in_col,in_row);
    //             printf("image_r:%f image_g:%f image_b:%f \n",*(choose_funtion(image,0,in_row,in_col,image_cols)),*(choose_funtion(image,1,in_row,in_col,image_cols)),*(choose_funtion(image,2,in_row,in_col,image_cols)));
    //         }
    //             index_++;
    //      }
    // }


    printf("run 2\n");
    int * data_bin = new int[667*825*3];
    printf("run 3\n");
    //double_to_int(after_affine,667*825*3,data_bin);
    printf("run 4\n");

    int ei=0;


    unsigned char * data_buffer = new unsigned char[667*825*3];
    int_to_unsignedchar(data_bin,data_buffer,667*825*3);

    for(int time=0;time<667*825*3;time++)
    {   if(data_buffer[time]!=0 && ei<100)
            {
                printf("data_bin:%x\n",(unsigned char )data_buffer[time]);
                ei++;
            }
    }


    save_data_to_local(after_affine,825*667*3,"affine");
    printf("run 5\n");







	return 0;
}
