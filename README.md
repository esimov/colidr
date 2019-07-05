# colidr
Coherent Line Drawing. 

Implementation of '[Coherent Line Drawing](http://umsl.edu/mathcs/about/People/Faculty/HenryKang/coon.pdf)' algorithm developed by Kang et al, NPAR 2007 in Go / [gocv](https://github.com/hybridgroup/gocv).

## Requirements
- Go 1.10 or higher, but it should work even with a lower version
- OpenCV 3
- gocv (bundled into the project, since it was extended with missing OpenCV functions needed for the implementation)

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

## Author

* Endre Simo ([@simo_endre](https://twitter.com/simo_endre))

## License
Copyright © 2019 Endre Simo

This project is under the MIT License. See the [LICENSE](https://github.com/esimov/triangle/blob/master/LICENSE) file for the full license text.
