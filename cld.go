package main

import "math"

func gauss(x, mean, sigma float64) float64 {
	return math.Exp((-(x - mean) * (x - mean)) / (2 * sigma * sigma)) / math.Sqrt(math.Pi * 2.0 * sigma * sigma)
}

func MakeGaussianVector(sigma float64, gau []float64) []float64 {
	var threshold = 0.001
	var i int

	for {
		i++
		if (gauss(float64(i), 0.0, sigma) < threshold) {
			break
		}
	}
	// clear slice
	gau = gau[:0]
	// extend slice
	gau = append(gau, make([]float64, i+1)...)
	gau[0] = gauss(0.0, 0.0, sigma)

	for j := 1; j < len(gau); j++ {
		gau[j] = gauss(float64(j), 0.0, sigma)
	}
	
	return gau
}