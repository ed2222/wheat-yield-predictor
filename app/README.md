These are the files required to deploy the wheat yield predictor app.

Make sure to have a Heroku account and Docker (https://docs.docker.com/get-started/get-docker/) up and running and to have access to the Heroku CLI (```pip install heroku```), then run the following commands in order :

1.   Log in to your Heroku account

```heroku login```

2.   Create the app

```heroku create [app name]```

3.   Set the Heroku stack to "container" to be able to run the Docker image

```heroku stack:set container -a [app name]```

4.   Push the app

```heroku container:push web -a [app name]```

5.   Release the app

```heroku container:release web -a [app name]```

6.   Open the Heroku app

```heroku open -a [app name]```
