
if [ ! -d dev ] ; then
  echo " - Download WMT data"
  wget -q http://www.statmt.org/wmt13/dev.tgz
  tar --wildcards -xf dev.tgz "dev/newstest2012.??"
  /bin/rm dev.tgz
fi