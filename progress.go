package colidr

import (
	"fmt"
	"time"
)

type event struct {
	done chan struct{}
}

func (e *event) start(msg string) {
	e.done = make(chan struct{}, 1)
	start := time.Now()
	ticker := time.NewTicker(time.Millisecond * 100)

	fmt.Printf("\t%s", msg)
	go func() {
		for {
			select {
			case <-ticker.C:
				fmt.Print(".")
			case <-e.done:
				ticker.Stop()
				end := time.Now().Sub(start)
				fmt.Printf("\nDone in: %.2fs\n", end.Seconds())
			}
		}
	}()
}

func (e *event) stop() {
	e.done <- struct{}{}
	time.Sleep(time.Millisecond)
}
