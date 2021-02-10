#!/bin/bash
set -e

VERSION="1.0.1"
PROTECTED_MODE="no"

export GO15VENDOREXPERIMENT=1
export GO111MODULE=on

cd $(dirname "${BASH_SOURCE[0]}")
OD="$(pwd)"

if [ "$(uname)" == "Darwin" ]; then
    export PKG_CONFIG_PATH="/usr/local/opt/opencv@3/lib/pkgconfig"
fi

# build and store objects into original directory.
go build -mod=vendor -ldflags "-X main.Version=$VERSION" -o "$OD/colidr" cli/main.go

echo $GOPATH
if [ -d $GOPATH ] ; then
    cp colidr $GOPATH/colidr
fi

rm $OD/colidr

