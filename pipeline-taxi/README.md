# Pipeline - taxi data validattion, training, analysis and serving

If you build image from behind corporate network, you may need to specify the proxy server using ```--build-arg```:

```
$ docker build --no-cache --build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy -t test_build_taxi:v0.1 .
```

