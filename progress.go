package colidr

import (
	"fmt"
	"os"
	"text/tabwriter"
	"time"
)

type event struct {
	msg  string
	done chan struct{}
}

var defaultMsg string

// newEvent constructor method for instantiating a new progress event.
func newEvent(msg string) *event {
	defaultMsg = msg
	return &event{msg: msg}
}

// start dispatch a new progress event
func (e *event) start() {
	e.done = make(chan struct{}, 1)
	start := time.Now()
	ticker := time.NewTicker(time.Millisecond * 100)

	w := tabwriter.NewWriter(os.Stdout, 10, 0, 0, ' ', tabwriter.DiscardEmptyColumns)
	fmt.Fprintf(w, "\r\t%s", e.msg)
	w.Flush()

	go func() {
		for {
			for _, r := range `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` {
				select {
				case <-ticker.C:
					w := tabwriter.NewWriter(os.Stdout, 10, 0, 0, ' ', tabwriter.DiscardEmptyColumns)
					fmt.Fprintf(w, "\r\t%s%s %c \t%s", e.msg, "\x1b[35m", r, "\x1b[39m")
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

// append appends a string to the existing one
func (e *event) append(msg interface{}) {
	e.msg += msg.(string)
}

// clear clears out the default message
func (e *event) clear() {
	e.msg = defaultMsg
}

// stop signals the process end
func (e *event) stop() {
	e.done <- struct{}{}
	time.Sleep(time.Millisecond)
}
