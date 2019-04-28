/**
   library structure
   $(RAW_HTML
   <object type="image/svg+xml" data="grain.svg" width="100%">
   )

   License: $(LINK2 http://boost.org/LICENSE_1_0.txt, Boost License 1.0).
   Authors:
   $(LINK2 https://github.com/ShigekiKarita, Shigeki Karita)
*/
module grain;

//    <img src="grain.svg", alt="dependency", width="100%">

public:
import grain.autograd;
import grain.chain;
import grain.config;
import grain.serializer;
import grain.optim;
import grain.metric;

version (grain_cuda) import grain.cuda;
