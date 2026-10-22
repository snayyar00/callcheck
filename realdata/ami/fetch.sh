#!/bin/sh
# Download the AMI annotations and the headset audio build_pairs.py uses (~2.2 GB, CC BY 4.0).
set -e
D=data/.corpora/ami
mkdir -p "$D/audio"
[ -d "$D/ann" ] || { curl -L -o "$D/ann.zip" https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
                     unzip -q "$D/ann.zip" -d "$D/ann"; }
for m in ES2002a ES2008c ES2012b IS1009b IS1003b TS3005a ES2010c ES2014b IS1008c; do
  for h in 0 1 2 3; do
    f="$D/audio/$m.Headset-$h.wav"
    [ -s "$f" ] || curl -L -o "$f" "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Headset-$h.wav"
  done
done
