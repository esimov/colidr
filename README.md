# colidr (Coherent Line Drawing)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat)](./LICENSE)
[![release](https://img.shields.io/badge/release-v1.0.1-blue.svg)](https://github.com/esimov/colidr/releases/tag/v1.0.1)

Golang implementation of '[Coherent Line Drawing](http://umsl.edu/mathcs/about/People/Faculty/HenryKang/coon.pdf)' algorithm developed by Kang et al, NPAR 2007.

![sample](https://user-images.githubusercontent.com/883386/60726045-40c83a80-9f43-11e9-9d53-7f190889e4bc.jpg)

## Requirements
- Go 1.10 or higher, but it should work even with a lower version
- OpenCV 3 (tested with 3.4.2)
- [gocv](https://github.com/hybridgroup/gocv) (bundled into the project, since it was extended with missing OpenCV functions needed for the implementation)
- [potrace](http://potrace.sourceforge.net/) - for transforming the bitmap into smooth, scalable image (however this is optional)

## Installation
```bash
$ go get -u github.com/esimov/colidr/
$ go install
```
Another option is cloning the repository and running the `make` file.
```bash
$ git clone https://github.com/esimov/colidr
$ cd colidr
$ make
```
This will generate the binary file.

## Usage
```bash
$ colidr -h

┌─┐┌─┐┬  ┬┌┬┐┬─┐
│  │ ││  │ ││├┬┘
└─┘└─┘┴─┘┴─┴┘┴└─

Coherent Line Drawing CLI
    Version: 1.0.1

  -aa
    	Anti aliasing
  -bl int
    	Blur size (default 3)
  -di int
    	Number of FDoG iteration
  -ei int
    	Number of Etf iteration (default 1)
  -in string
    	Source image
  -k int
    	Etf kernel (default 3)
  -out string
    	Destination image
  -pt
    	Use potrace to smooth edges (default true)
  -rho float
    	Rho (default 0.98)
  -sc float
    	SigmaC (default 1)
  -sm float
    	SigmaM (default 3)
  -sr float
    	SigmaR (default 2.6)
  -tau float
    	Tau (default 0.98)
  -ve
    	Visualize Etf
  -vr
    	Visualize end result

```
Feel free to play with the values in order to modify the visual output of the generated (non-photorealistically rendered) image. To obtain higher fidelity results you need to increase the `kernel` value and also the ETF iteration number. Different combinations produces completely different output. The `-di`, `-ei`, `-k` flags are mostly used for fine tuning, on the other hand `-rho` and `-tau` flags could change dramatically the rendered output.

You can also visualize the edge tangent flow if you enable the `-ve` flag. Below is the process illustrated:

| Original image | Edge tangent flow | Coherent line drawing (final output)
|:--:|:--:|:--:|
| ![original](https://user-images.githubusercontent.com/883386/60724812-0f9a3b00-9f40-11e9-86c2-906bc652b3f6.jpg) | ![flowfield](https://user-images.githubusercontent.com/883386/60726316-ea0f3080-9f43-11e9-9b6c-c9bac05b32f0.png) | ![output](https://user-images.githubusercontent.com/883386/60725818-b1228c00-9f42-11e9-9019-6280d31aa09f.png) | 

Using the `-pt` flag you can trace the generated bitmap into a smooth scalabe image. You need to have [potrace](http://potrace.sourceforge.net/) installed on your machine for this scope.

Below is an example whith and without the potrace flag activated.

| Normal output | Potrace activated
|:--:|:--:|
| ![normal](https://user-images.githubusercontent.com/883386/60726045-40c83a80-9f43-11e9-9d53-7f190889e4bc.jpg) | ![smooth](https://user-images.githubusercontent.com/883386/60726046-40c83a80-9f43-11e9-81b8-d98bfea90991.jpg) |

The above [image](http://hof.povray.org/images/patio.jpg) was ganareted with the following command:

```bash
colidr -in ~/Desktop/patio.jpg -out ~/Desktop/patio_scene.png -k=1 -sr=2.5 -sm=3.2 -tau=0.9975 -di=1 -aa=1 -ve=1 -vr=0 -pt=1 -ei=1
```

## Sample images
| Rasterized bitmap | Vectorized image
|:--:|:--:|
| ![great_wave_of_kanagawa](https://user-images.githubusercontent.com/883386/60795446-5cfeee00-a174-11e9-8f55-08a3695eca56.png) | ![great_wave_of_kanagawa](https://user-images.githubusercontent.com/883386/60795445-5cfeee00-a174-11e9-9c7a-492fb72b3f69.png) |
| ![starry_night](https://user-images.githubusercontent.com/883386/60795440-5c665780-a174-11e9-9804-d5e56d0c49e7.png) | ![starry_night](https://user-images.githubusercontent.com/883386/60795439-5c665780-a174-11e9-9d01-2c05373b465e.png) |
| ![people](https://user-images.githubusercontent.com/883386/60795438-5c665780-a174-11e9-8c8a-365bd8eda329.png) | ![people](https://user-images.githubusercontent.com/883386/60795436-5bcdc100-a174-11e9-8fe2-00e6a0f13f06.png) |
| ![tiger](https://user-images.githubusercontent.com/883386/60795443-5cfeee00-a174-11e9-9fd4-6ceb9a02ca21.png) | ![tiger](https://user-images.githubusercontent.com/883386/60795442-5c665780-a174-11e9-922b-fd5517622157.png) |

## Todo
- [ ] GUI support

## Author

* Endre Simo ([@simo_endre](https://twitter.com/simo_endre))

## License
Copyright © 2019 Endre Simo

This project is under the MIT License. See the [LICENSE](https://github.com/esimov/colidr/blob/master/LICENSE) file for the full license text.
