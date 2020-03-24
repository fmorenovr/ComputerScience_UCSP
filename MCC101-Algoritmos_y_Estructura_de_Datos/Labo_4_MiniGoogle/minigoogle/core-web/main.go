package main

import (
  "time";
  "bytes";
  "strconv";
  "strings";
  "net/http";
  "io/ioutil";
  "encoding/json";
  "html/template";
  "github.com/gorilla/mux";
  "github.com/jenazads/gologger";
  "github.com/jenazads/gojwt";
)

type DataFound struct{
  Id      string   `json:"id"`
  Title   string   `json:"title"`
  Content string   `json:"content"`
}

type SearchQuery struct{
  State    int     `json:"state"`
  Query    string  `json:"query"`
}

type NewPageQuery struct{
  State      int  `json:"state"`
  IdRequest  int  `json:"idRequest"`
  NroPage    int  `json:"nroPage"`
}

type FinishQuery struct{
  State      int  `json:"state"`
  IdRequest  int  `json:"idRequest"`
}

type Data struct {
  Result     []DataFound  `json:"result"`
  IdRequest  int          `json:"idRequest"`
  NroPages   int          `json:"nroPages"`
}

type DataToWeb struct {
  Result     []DataFound  `json:"result"`
  IdRequest  int          `json:"idRequest"`
  NroPages   []int        `json:"nroPages"`
}

var DataPages DataToWeb
var AuxIdRequest int = -1
var WasQuery bool = false
var SearchText string
var pageInfo DataFound

const(
// hostnames
  Hostname = "pocosearch.fmorenovr.com"//"minigoogle.gescloud.io"//"localhost"
// http
  Httpprotocol   = "http://"
  ListenHTTP     = ":80"
// index paths
  Template_index      = "index.html"
  Template_about      = "templates/about.html"
  Template_pageInfo   = "templates/pageinfo.html"
  Template_pagesfound = "templates/pagesfound.html"
  Template_notFound   = "templates/notfound.html"
// directories
  css = "assets/css"
  depen = "assets/dependencies"
  font = "assets/fonts"
  js = "assets/js"
  img = "assets/images"
  
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
  
  cssStrip = http.StripPrefix("/assets/css/", cssHandler)
  depenStrip = http.StripPrefix("/assets/dependencies/", depenHandler)
  fontStrip = http.StripPrefix("/assets/fonts/", fontHandler)
  jsStrip = http.StripPrefix("/assets/js/", jsHandler)
  imgStrip = http.StripPrefix("/assets/images/", imgHandler)
)

type Page struct {
  Title   string
  Body  []byte
}

func makeRange(min, max int) []int {
  a := make([]int, max-min+1)
  for i := range a {
    a[i] = min + i
  }
  return a
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

func FromJSON(data []byte) (DataToWeb, error) {
  elements := make(map[string]Data)
  err := json.Unmarshal(data, &elements)
  return DataToWeb{Result: elements["data"].Result, IdRequest: elements["data"].IdRequest, NroPages: makeRange(1,elements["data"].NroPages)}, err
}

func searchById(v string, data []DataFound) ([]DataFound){
  var datos []DataFound = nil
  for _, elem := range data {
    if v == elem.Id{
      datos = append(datos, elem)
      return datos
    }
  }
  return nil
}

func SearchHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/plain; text/html; charset=utf-8")
  r.ParseForm()
  start := time.Now()
  SearchText = strings.ToLower(r.Form["search-text"][0])
  sq := SearchQuery{State: 0, Query: SearchText}
  jsonObj, _ := gojwt.ToJSON(sq)
  req, err := http.NewRequest("POST", "http://"+Hostname+":8090/search",  bytes.NewBuffer(jsonObj))
  if err!=nil{
    Logger.Error("Error: %s", err)
    return
  }
  req.Header.Set("Content-Type", "application/json;application/x-www-form-urlencoded; charset=utf-8")
  client := &http.Client{}
  resp, err := client.Do(req)
  if err!=nil{
    Logger.Error("Error: %s", err)
    return
  }
  defer resp.Body.Close()
  body, _ :=ioutil.ReadAll(resp.Body)
  DataPages, _ = FromJSON([]byte(body))
  LogServer(r.Method, r.URL.Path,"Search")
  Logger.Info("method: %s", r.Method)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
  http.Redirect(w, r, "/pagesfound", http.StatusMovedPermanently)
}

func SearchPageHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/html")
  r.ParseForm()
  start := time.Now()
  v, _ := strconv.ParseInt(r.Form["number"][0], 10, 64)
  pageNum := int(v)
  sq := NewPageQuery{State: 1, IdRequest: AuxIdRequest, NroPage: pageNum}
  jsonObj, _ := gojwt.ToJSON(sq)
  req, err := http.NewRequest("POST", "http://"+Hostname+":8090/search",  bytes.NewBuffer(jsonObj))
  if err!=nil{
    Logger.Error("Error: %s", err)
    return
  }
  req.Header.Set("Content-Type", "application/json;application/x-www-form-urlencoded")
  client := &http.Client{}
  resp, err := client.Do(req)
  if err!=nil{
    Logger.Error("Error: %s", err)
    return
  }
  defer resp.Body.Close()
  body, _ :=ioutil.ReadAll(resp.Body)
  DataPages, _ = FromJSON([]byte(body))
  LogServer(r.Method, r.URL.Path,"Search")
  Logger.Info("method: %s", r.Method)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
  http.Redirect(w, r, "/pagesfound", http.StatusMovedPermanently)
}

func IndexHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/html")
  if WasQuery == true{
    sq := FinishQuery{State: 2, IdRequest: AuxIdRequest}
    jsonObj, _ := gojwt.ToJSON(sq)
    req, err := http.NewRequest("POST", "http://"+Hostname+":8090/search",  bytes.NewBuffer(jsonObj))
    if err!=nil{
      Logger.Error("Error: %s", err)
      return
    }
    req.Header.Set("Content-Type", "application/json;application/x-www-form-urlencoded")
    client := &http.Client{}
    resp, err := client.Do(req)
    if err!=nil{
      Logger.Error("Error: %s", err)
      return
    }
    defer resp.Body.Close()
    WasQuery = false
  }
  if AuxIdRequest != 0 {
    AuxIdRequest = 0
  }
  start := time.Now()
  LogServer(r.Method, r.URL.Path,"Index")
  p, _ := LoadPage(Template_index)
  w.Write(p.Body)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
}

func AboutHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/html")
  start := time.Now()
  LogServer(r.Method, r.URL.Path,"About")
  t, _ := template.ParseFiles(Template_about)
  t.Execute(w, nil)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
}

func PagesFoundHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/html")
  AuxIdRequest = DataPages.IdRequest
  WasQuery = true
  start := time.Now()
  LogServer(r.Method, r.URL.Path,"Pages Found")
  t, _ := template.ParseFiles(Template_pagesfound)
  t.Execute(w, DataPages)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
}

func NotFoundHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/html")
  start := time.Now()
  LogServer(r.Method, r.URL.Path,"NotFound")
  t, _ := template.ParseFiles(Template_notFound)
  t.Execute(w, nil)
  Logger.Info("Completed %s in %v\n", r.URL.Path, time.Since(start))
}

func PageInfoHandler(w http.ResponseWriter, r *http.Request) {
  w.Header().Set("Content-Type", "text/html")
  r.ParseForm()
  start := time.Now()
  v := r.Form["idpage"][0]
  pageInfo := searchById(v, DataPages.Result)
  LogServer(r.Method, r.URL.Path,"Page Info")
  t, _ := template.ParseFiles(Template_pageInfo)
  t.Execute(w, pageInfo[0])
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
  router.HandleFunc("/about", AboutHandler).Methods("GET")
  router.HandleFunc("/pagesfound", PagesFoundHandler)
  router.HandleFunc("/search", SearchHandler).Methods("GET")
  router.HandleFunc("/searchpage", SearchPageHandler).Methods("GET")
  router.HandleFunc("/pageinfo", PageInfoHandler).Methods("GET")
  
  router.NotFoundHandler = http.HandlerFunc(NotFoundHandler)
  
  muxHttp := http.NewServeMux()
  muxHttp.Handle("/", router)
  
  // directories
  muxHttp.Handle("/assets/css/", cssStrip)
  muxHttp.Handle("/assets/dependencies/", depenStrip)
  muxHttp.Handle("/assets/fonts/", fontStrip)
  muxHttp.Handle("/assets/js/", jsStrip)
  muxHttp.Handle("/assets/images/", imgStrip)
  
  MuxInitService(muxHttp)
}

func main() {
  HttpListenerServiceInit()
}

