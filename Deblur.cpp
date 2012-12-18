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

Color intensity(Image<Color,2>& I, int i, int j){
	if(i<0 || i>=I.width() || j<0 || j>=I.height()){
		return Color(0,0,0);
	}
	return I(i,j);
}

float* img2floats(Image<Color,2>& I){
	int i, j, w = I.width(), h = I.height();
	float* If = new float[w*h];

	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			If[i+j*w] = I(i,j);
		}
	}
	
	return If;
}

float* img2megafloats(Image<Color,2>& I, int kernel_size){
	int i, j, ii, jj, w = I.width(), h = I.height();
	float* Imf = new float[w*h*kernel_size];

	for(i=0; i<w; ++i){
		for(j=0; j<h; ++j){
			for(ii=-kernel_size/2; ii<=kernel_size/2; ++ii){
				for(jj=-kernel_size; jj<=kernel_size/2; ++jj){
					Imf[i+j*w] = intensity(I, i+ii, j+jj);
				}
			}
		}
	}
	
	return Imf;
}

float* kernelEstimation(){

}

void deblur(Image<Color,2>& I, Image<Color,2>& B, Image<Color,2>& Nd){
	int i, niter;

	for(i = 0; i<niter; ++i){

	}
	
}

int main()
{
	// Load and display images
	Image<Color,2> B, Nd, I;
	if( ! load(B, srcPath("lena_blurred.jpg")) ||
        ! load(Nd, srcPath("lena_denoised.jpg")) ||
		! load(I, srcPath("lena_denoised.jpg")) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
	int w = B.width();
	int h = B.height();
	
	// display window
	openWindow(2*w, h);
	display(B,0,0);
	display(Nd,w,0);	

	endGraphics();
	return 0;
}
