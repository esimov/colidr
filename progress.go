package colidr

import (
	"fmt"
	"os"
	"text/tabwriter"
	"time"
)

type event struct {
	done chan struct{}
}

func (e *event) start(msg string) {
	e.done = make(chan struct{}, 1)
	start := time.Now()
	ticker := time.NewTicker(time.Millisecond * 100)

	w := tabwriter.NewWriter(os.Stdout, 10, 0, 0, ' ', tabwriter.DiscardEmptyColumns)
	fmt.Fprintf(w, "\r\t%s", msg)
	w.Flush()

	go func() {
		for {
			for _, r := range `-\|/` {
				select {
				case <-ticker.C:
					w := tabwriter.NewWriter(os.Stdout, 10, 0, 0, ' ', tabwriter.DiscardEmptyColumns)
					fmt.Fprintf(w, "\r\t%s%s %c\t%s", msg, "\x1b[92m", r, "\x1b[39m")
					w.Flush()
				case <-e.done:
					ticker.Stop()
					w := tabwriter.NewWriter(os.Stdout, 20, 15, 10, '.', tabwriter.AlignRight|tabwriter.DiscardEmptyColumns)
					end := time.Now().Sub(start)
					fmt.Fprintf(w, "\t%.2fs\t\n", end.Seconds())
					w.Flush()
				}
			}
		}
	}()
}

func (e *event) stop() {
	e.done <- struct{}{}
	time.Sleep(time.Millisecond)
}
