package colidr

import "math"

// gauss computes the gaussian function of variance
func gauss(x, mean, sigma float64) float64 {
	return math.Exp((-(x-mean)*(x-mean))/(2*sigma*sigma)) / math.Sqrt(math.Pi*2.0*sigma*sigma)
}

// makeGaussianVector constructs a gaussian vector field of floats
func makeGaussianVector(sigma float64) []float64 {
	var (
		gau       []float64
		threshold = 0.001
		i         int
	)

	for {
		i++
		if gauss(float64(i), 0.0, sigma) < threshold {
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

func abs(val float32) float32 {
	if val < 0.0 {
		return -val
	}
	return val
}

// absInt return the absolute value of x
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
