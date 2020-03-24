package main

import (
  "time";
  "net/http";
  "io/ioutil";
  "github.com/gorilla/mux";
  "github.com/jenazads/logs"
)

const(
// hostnames
  Hostname = "localhost"
// http
  Httpprotocol   = "http://"
  ListenHTTP     = ":80"
// index paths
  Template_index = "index.html"
// directories
  css = "css"
  depen = "dependencies"
  font = "font"
  js = "js"
  img = "images"
  
  CSS = http.Dir(css)
  Depen = http.Dir(depen)
  Font = http.Dir(font)
  JS = http.Dir(js)
  IMG = http.Dir(img)
)

var(
  Logger = logs.NewLogger(10000)
  cssHandler = http.FileServer(CSS)
  depenHandler = http.FileServer(Depen)
  fontHandler = http.FileServer(Font)
  jsHandler = http.FileServer(JS)
  imgHandler = http.FileServer(IMG)
  
  cssStrip = http.StripPrefix("/css/", cssHandler)
  depenStrip = http.StripPrefix("/dependencies/", depenHandler)
  fontStrip = http.StripPrefix("/font/", fontHandler)
  jsStrip = http.StripPrefix("/js/", jsHandler)
  imgStrip = http.StripPrefix("/images/", imgHandler)
)

type Page struct {
  Title   string
  Body  []byte
}

func VerifyError(err error){
  if err !=nil {
    Logger.Error("%s",err)
    panic(err)
  }
}

// load page
func LoadPage(filename string) (*Page, error) {
  body, err := ioutil.ReadFile(filename)
  if err != nil {
    return nil, err
  }
  return &Page{Title: filename, Body: body}, nil
}

func LogServer(method, path, name string){
  Logger.Info("Started %s %s", method, path)
  Logger.Info("Executing BeaGons "+name+" Handler")
}

func IndexHandler(w http.ResponseWriter, r *http.Request) {
  start := time.Now()
  LogServer(r.Method, r.URL.Path,"Index")
  p, _ := LoadPage(Template_index)
  w.Write(p.Body)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
}


func MuxInitService(muxHttp *http.ServeMux){
  server := &http.Server{
    Addr   : ListenHTTP,
    Handler: muxHttp,
  }
  Logger.Info("webAppClient-Go Service Start")
  Logger.Info("listening on %s%s%s",Httpprotocol,Hostname,ListenHTTP)
  err := server.ListenAndServe()
  //err := http.ListenAndServe(ListenHTTP, nil)
  VerifyError(err)
}

func HttpListenerServiceInit(){
  // router
  router := mux.NewRouter()
  router.HandleFunc("/", IndexHandler)
  //router.NotFoundHandler = http.HandlerFunc(handlers.NotFoundHandler)
  
  muxHttp := http.NewServeMux()
  muxHttp.Handle("/", router)
  
  // directories
  muxHttp.Handle("/css/", cssStrip)
  muxHttp.Handle("/dependencies/", depenStrip)
  muxHttp.Handle("/font/", fontStrip)
  muxHttp.Handle("/js/", jsStrip)
  muxHttp.Handle("/images/", imgStrip)
  
  MuxInitService(muxHttp)
}

func main() {
  HttpListenerServiceInit()
}

