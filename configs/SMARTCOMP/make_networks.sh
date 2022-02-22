# Grid Networks
netgenerate --grid --grid.number=3 --grid.length=100 --no-turnarounds -j=traffic_light --output=grid-3x3.net.xml --grid.attach-length=100 -L=3 --no-left-connections

netgenerate --grid --grid.number=5 --grid.length=100 --no-turnarounds -j=traffic_light --output=grid-5x5.net.xml --grid.attach-length=100 -L=3 --no-left-connections

netgenerate --grid --grid.number=7 --grid.length=100 --no-turnarounds -j=traffic_light --output=grid-7x7.net.xml --grid.attach-length=100 -L=3 --no-left-connections


# Spider Network
# netgenerate --spider --spider.arm-number=4 --spider.circle-number=3 --spider.omit-center -j=traffic_light -L=3 --spider.space-radius=150 --no-turnarounds --output=spider.net.xml


# Random Network
# netgenerate --rand --rand.iterations=20 --rand.random-lanenumber=true -L=2 -j=traffic_light --no-turnarounds --output=rand.net.xml