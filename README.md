# temp_code
I'm going to use this repository to hold temporary code and data. The master branch will be empty, but different branches will hold different things.

## NR simulation points choosing

This branch gives some example code for generating NR simulation points. The process is to get some pe posteriors (these are propietary, so no example here, sorry!) Then run:

```
posterior_to_90pt_xml.py
source runme.sh
source runme_2.sh
```

To run the 3 code steps in order.

This is still pretty rough, and some functionality that presumably would be needed (for e.g. incorporating NR simulations that already exist) is not here. However, this can be added, but I don't want to waste my time if this is not useful to anyone!
