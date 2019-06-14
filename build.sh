#!/bin/bash
set -e

VERSION="1.0.1"
PROTECTED_MODE="no"

export GO15VENDOREXPERIMENT=1
export GO111MODULE=on

cd $(dirname "${BASH_SOURCE[0]}")
OD="$(pwd)"

# build and store objects into original directory.
go build -mod=vendor -ldflags "-X main.Version=$VERSION" -o "$OD/colidr" cli/main.go

if [ -d $GOPATH ]
then
    cp colidr $GOPATH/bin
else
    cp colidr /usr/local/bin
fi

rm $OD/colidr

