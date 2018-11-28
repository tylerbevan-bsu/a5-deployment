# a5-deployment
Tyler Bevan

App is hosted at https://sentiment-cs533-tylerbevan.herokuapp.com/ and the source code is on github at 
https://github.com/tylerbevan-bsu/a5-deployment/

I ended up using the external Flask-Session library to create server-side sessions so that each client gets their own
instance of the model instead of a single global one. Because the web server is happy to spin off multiple server
processes this worked better. I also put all of the actual pandas code in its own file to keep it separate from the
Flask driving code.

