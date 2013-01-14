// Imagine++ project
// Project:  Deblur
// Author:   Yohann Salaun
// Date:     2012/12/25

#include <Imagine/Images.h>
#include <Imagine/Graphics.h>
#include <vector>
#include <Imagine/LinAlg.h>

using namespace Imagine;
using namespace std;

// intensity functions ---------------------------------------
Color intensity(const Image<Color,2>& I, int i, int j){
	if(i<0 || i>=I.width() || j<0 || j>=I.height()){
		return Color(0,0,0);
	}
	return I(i,j);
}

float color2gray(Color c){
	return (c.r()/3.0+c.g()/3.0+c.b()/3.0)/256.0;
}

// display functions -----------------------------------------
void displayMatrix(Matrix<float> A, int w, int h){
	int i,j;
	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			cout << A[i+j*w] << " ";
		}
		cout << endl;
	}
}

void displayVector(Vector<float> A){
	int i;
	for(i=0; i<A.size(); ++i){
		cout << A[i] << endl;
	}
}

void displayKernel(Vector<float> k, int k_size){
	int i,j;
	for(i=0; i<k_size; ++i){
		for(j=0; j<k_size; ++j){
			cout << k[i+j*k_size] << " ";
		}
		cout << endl;
	}
}

// conversions functions ------------------------------------
Vector<float> img2vect(const Image<Color,2>& I){
	int i, j, w = I.width(), h = I.height();
	Vector<float> If;
	If.setSize(w*h);

	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			If[i+j*w] = color2gray(I(i,j));
		}
	}
	
	return If;
}

Image<Color,2> vect2img(Vector<float>& I, int w, int h){
	int i, j, c;
	Image<Color,2> Ic(w, h);

	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			c = int(256*I[i+j*w]);
			Ic(i,j) = Color(c, c, c);
		}
	}
	
	return Ic;
}

Matrix<float> img2kernelMatrix(const Image<Color,2>& I, const int kernel_size){
	int i, j, ii, jj, w = I.width(), h = I.height();
	Matrix<float> Imf;
	Imf.setSize(w*h, kernel_size*kernel_size);
	int ks = kernel_size/2;

	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			for(ii=-ks; ii<=ks; ++ii){
				for(jj=-ks; jj<=ks; ++jj){
					Imf(i+j*w, ii+ks+(jj+ks)*kernel_size) = color2gray(intensity(I, i+ii, j+jj));
				}
			}
		}
	}
	
	return Imf;
}

// convolution with kernel function -------------------------
Image<Color,2> kernelBlurring(Image<Color,2>& I, Vector<float> k, const int kernel_size){
	int i, j, ii, jj;
	int w = I.width(), h = I.height();
	int ks = kernel_size/2;
	Color c;
	float r, g, b;
	Image<Color,2> Ik(w, h);

	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			r = g = b = 0;
			for(ii=-ks; ii<=ks; ++ii){
				for(jj=-ks; jj<=ks; ++jj){
					c = intensity(I, i+ii, j+jj);
					r += c.r()*k[ii+ks+(jj+ks)*kernel_size];
					g += c.g()*k[ii+ks+(jj+ks)*kernel_size];
					b += c.b()*k[ii+ks+(jj+ks)*kernel_size];
				}
			}
			Ik(i,j) = Color(r, g, b);
		}
	}
	return Ik;
}


// algorithm ------------------------------------------------

// projection for kernel estimation part
void projection(Vector<float>& k){
	int i;
	float norm = 0;

	for(i=0; i<k.size(); ++i){
		k[i] = max(k[i], float(0));
		norm += k[i];
	}
	if(norm > 0)
	{
		for(i=0; i<k.size(); ++i){
			k[i] /= norm;
		}
	}
}

// kernel estimation
Vector<float> kernelEstimation(const Image<Color,2>& I, const Image<Color,2>& B, const int kernel_size){
	int i, j;
	float k_size2 = kernel_size*kernel_size;
	float norm;

	// parameters
	int niter = 100000;						// number of iterations
	float beta = 0.000001;					// parameter for reccursive formula of k
	float lambda = 1;						// smoothness parameter
	float lambda2 = lambda*lambda;
	
	// Vector_Matrix form
	Vector<float> b = img2vect(B);
	Vector<float> k;
	k.setSize(kernel_size*kernel_size);
	Matrix<float> A = img2kernelMatrix(I, kernel_size);
	Matrix<float> AtA = transpose(A)*A;
	Vector<float> Atb = transpose(A)*b;

	// initialize k as a dirac
	for(i=0; i<kernel_size*kernel_size; ++i){
		k[i] = 0;
	}
	k[kernel_size/2*(1 + kernel_size)] = 1;

	// loop to find k
	for(i=0; i<niter; ++i){
		k = k + beta*(Atb -	AtA*k - lambda2*k);
	}

	return k;
}

// deconvolution algorithm
void deconvol(Image<Color,2>& I, Image<Color,2>& B, Image<Color,2>& Nd, Vector<float> k){
	int k_size = int(sqrt(float(k.size())));
	int niter = 10;
	int i;

	// initialization of delta_pictures
	Image<Color,2> deltaI = I - Nd;
	Image<Color,2> deltaB = B - kernelBlurring(Nd, k, k_size);
	
	for(i = 0; i<niter; ++i){

	}

}

// main algorithm
void deblur(Image<Color,2>& I, const Image<Color,2>& B, const Image<Color,2>& Nd, int k_size){
	int i;
	int niter = 1;

	for(i = 0; i<niter; ++i){
		// estimate kernel with current picture
		Vector<float> k = kernelEstimation(I, B, k_size);

		// deconvolution of blurred picture

	}
	
}

// main -----------------------------------------------------
int main()
{
	// Load and display images
	Image<Color,2> B, Nd, I;
	if( ! load(B, srcPath("lena_blurred.jpg")) ||
        ! load(Nd, srcPath("lena_denoised.jpg")) ||
		! load(I, srcPath("lena.jpg")) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
	int w = B.width();
	int h = B.height();

	int k_size = 5;

	// create blur kernel
	int ii, jj;
	Vector<float> kernel;
	kernel.setSize(k_size*k_size);
	int ks = k_size/2;
	float norm = 0;
	
	// initialize k as a gaussian
	for(ii=-ks; ii<=ks; ++ii){
		for(jj=-ks; jj<=ks; ++jj){
			kernel[ii+ks+(jj+ks)*k_size] = rand();
			kernel[ii+ks+(jj+ks)*k_size] = exp(-float(ii*ii+jj*jj));
			norm += kernel[ii+ks+(jj+ks)*k_size];
		}
	}
	kernel /= norm;

	// compute blurred picture
	B = kernelBlurring(I, kernel, k_size);

	// estimate kernel
	Vector<float> k = kernelEstimation(I, B, k_size);

	// display kernels
	// real kernel
	displayKernel(kernel, k_size);
	// estimated kernel
	cout << "------------------" << endl;
	projection(k);
	displayKernel(k, k_size);
	cout << "------------------" << endl;
	// norm 2 of difference
	cout << norm2(k-kernel)/norm2(kernel) << endl;
	cout << "------------------" << endl;

	// display window
	openWindow(4*w, h);
	Image<Color,2> A = kernelBlurring(I, kernel, k_size);
	Image<Color,2> C = kernelBlurring(I, k, k_size);
	display(A,0,0);
	display(B,w,0);
	display(C,2*w,0);
	display(I,3*w,0);	

	endGraphics();
	return 0;
}
