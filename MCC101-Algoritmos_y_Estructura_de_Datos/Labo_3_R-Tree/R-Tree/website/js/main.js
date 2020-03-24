var min = 99;
var max = 999999;
var polygonMode = false;
var pointMode = false;
var rangeSearchMode = false;
var knnSearchMode = false;
var pointArray = new Array();
var lineArray = new Array();
var activeLine;
var activeShape;
var canvas;
var poligonos = [];
var polyToRender = [];
var polCount = 0;
var rect1, isDown, origX, origY;//para el rectangulo
var RSxmin, RSymin, RSxmax, RSymax;
var knnIsCheck = document.getElementById("knn-search");
var colorFounded = "#2eb82e"
var MBR = [];

//------------------------------------------------------------
$(window).load(function(){
  prototypefabric.initCanvas();
  $('#create-polygon').click(function() {
    polygonMode = true;
    pointMode = false;
    rangeSearchMode = false;
    knnSearchMode = false;
    prototypefabric.polygon.drawPolygon();
    knnIsCheck.checked = false;
    showKNNOptions()
    RepaintCanvas(MBR);
  });
 $('#create-point').click(function() {
    polygonMode = false;
    pointMode = true;
    rangeSearchMode = false;
    knnSearchMode = false;
    prototypefabric.point.drawPoint();
    knnIsCheck.checked = false;
    showKNNOptions()
    RepaintCanvas(MBR);
  });
 $('#range-search').click(function() {
    polygonMode = false;
    pointMode = false;
    rangeSearchMode = true;
    knnSearchMode = false;
    knnIsCheck.checked = false;
    showKNNOptions()
    RepaintCanvas(MBR);
  });
  $('#knn-search').click(function(){
    polygonMode = false;
    pointMode = false;
    rangeSearchMode = false;
    knnSearchMode = true;
    RepaintCanvas(MBR);
  });
});

var prototypefabric = new function(){
  this.initCanvas = function () {
    canvas = window._canvas = new fabric.Canvas('c');
    canvas.setWidth(800);
    canvas.setHeight(600);
    //canvas.selection = false;

    canvas.on('mouse:down', function (options) {
      if(options.target && options.target.id == pointArray[0].id){
        prototypefabric.polygon.generatePolygon(pointArray);
        poligonos[polCount] = prototypefabric.polygon.polygonPoints;
        var minmaxPoints = minmax_pol(prototypefabric.polygon.polygonPoints)
        var toSend = {
          order: polCount,
          minP: [minmaxPoints[0][0], minmaxPoints[0][1]],
          maxP: [minmaxPoints[1][0], minmaxPoints[1][1]],
        }
        mqttPublish(local_clientMQTTPaho, "web/insert", toSend)
        prototypefabric.polygon.polygonPoints = [];
        prototypefabric.polygon.polygonLength = 0;
        polCount++;
      }
      if(polygonMode){
        prototypefabric.polygon.addPoint(options);
      }
      if(knnSearchMode && knnIsCheck.checked){
        var knn = document.getElementsByName("quantity")[0].value;
        var toSend = {
          x: parseInt(canvas.getPointer().x),
          y: parseInt(canvas.getPointer().y),
          k: parseInt(knn),
        }
        RepaintCanvas(MBR);
        if(poligonos.length >0){
          mqttPublish(local_clientMQTTPaho, "web/knn", toSend)
        }
      }
      if(pointMode){
        prototypefabric.point.addPoint(options);
        poligonos[polCount] = prototypefabric.point.points;
        var toSend = {
          order: polCount,
          minP: [prototypefabric.point.points[0], prototypefabric.point.points[1]],
          maxP: [prototypefabric.point.points[0], prototypefabric.point.points[1]],
        }
        mqttPublish(local_clientMQTTPaho, "web/insert", toSend)
        prototypefabric.point.points = [];
        polCount++;
      }
      if(rangeSearchMode){
        isDown = true;
        var pointer = canvas.getPointer(options.e);
        origX = pointer.x;
        origY = pointer.y;
        var pointer = canvas.getPointer(options.e);
        rect1 = new fabric.Rect({
          left: origX,
          top: origY,
          originX: 'left',
          originY: 'top',
          width: pointer.x-origX,
          height: pointer.y-origY,
          angle: 0,
          evented: false,
          fill: 'transparent',
          strokeWidth: 2,
          strokeDashArray: [10, 5],
          stroke: 'black',
        });
        canvas.add(rect1);
      }
    });

    canvas.on('mouse:up', function (options) {
      if(rangeSearchMode){
        isDown = false;
        RSxmin = parseInt(origX);
        RSymin = parseInt(origY);
        
        if(RSxmin > RSxmax){
          var temp = RSxmin
          RSxmin = RSxmax
          RSxmax = temp
        }
        if(RSymin > RSymax){
          var temp = RSymin
          RSymin = RSymax
          RSymax = temp
        }
        var toSend={
          minP:[RSxmin, RSymin],
          maxP:[RSxmax, RSymax],
        }
        if(poligonos.length >0){
          mqttPublish(local_clientMQTTPaho, "web/search", toSend)
        }
        rangeSearchMode = false;
      }
    });
    canvas.on('mouse:move', function (options) {
      if(rangeSearchMode){
        if (!isDown) return;
        var pointer = canvas.getPointer(options.e);
        if(origX>pointer.x){
          rect1.set({ left: Math.abs(pointer.x) });
        }
        if(origY>pointer.y){
          rect1.set({ top: Math.abs(pointer.y) });
        }
        rect1.set({ width: Math.abs(origX - pointer.x) });
        rect1.set({ height: Math.abs(origY - pointer.y) });
        canvas.renderAll();
        RSxmax = parseInt(pointer.x);
        RSymax = parseInt(pointer.y);
      }
      if(activeLine && activeLine.class == "line"){
        var pointer = canvas.getPointer(options.e);
        activeLine.set({ x2: pointer.x, y2: pointer.y });

        var points = activeShape.get("points");
        points[pointArray.length] = {
          x:pointer.x,
          y:pointer.y
        }
        activeShape.set({
          points: points
        });
        canvas.renderAll();
      }
      //------desactiva los  objetos para editarse-----
      canvas.deactivateAll();
      canvas.selection = false;
      canvas.forEachObject(function(o) {
      o.selectable = false;
       });
     //------------------------------------------------- 
      canvas.renderAll();
    });
  };
};

function deleteObjects(){
  var activeObject = canvas.getActiveObject(),
  activeGroup = canvas.getActiveGroup();
  if (activeObject) {
    if (confirm('Are you sure?')) {
      canvas.remove(activeObject);
    }
  }
  else if (activeGroup) {
    if (confirm('Are you sure?')) {
      var objectsInGroup = activeGroup.getObjects();
      canvas.discardActiveGroup();
      objectsInGroup.forEach(function(object) {
      canvas.remove(object);
      });
    }
  }
};

function resize() {
  var canvasSizer = document.getElementById("pizarra");
  var canvasScaleFactor = canvasSizer.offsetWidth/525;
  var width = canvasSizer.offsetWidth;
  var height = canvasSizer.offsetHeight;
  var ratio = canvas.getWidth() /canvas.getHeight();
  if((width/height)>ratio){
    width = height*ratio;
  } else {
    height = width / ratio;
  }
  var scale = width / canvas.getWidth();
  var zoom = canvas.getZoom();
  zoom *= scale;   
  canvas.setDimensions({ width: width, height: height });
  canvas.setViewportTransform([zoom , 0, 0, zoom , 0, 0])
}

window.addEventListener('load', resize, false);
window.addEventListener('resize', resize, false);

function RepaintCanvas(regiones){
  canvas.clear();
  RepaintPoly();
  dibujarMBR(regiones);
}

function RepaintPoly(){
  for (var i = 0; i < polyToRender.length ; i++) {
    canvas.add(polyToRender[i].setColor("red"))
  }
}

function dibujarMBR(regiones){
  var color="blue";
  for (var i = 0; i < regiones.length ; i++) {
    if( i>0 && parseInt(regiones[i].nivel) != parseInt(regiones[i-1].nivel)){
      color=getRandomColor();
    }
    var xmax=parseInt(regiones[i].maxP[0]);
    var ymax=parseInt(regiones[i].maxP[1]);
    var xmin=parseInt(regiones[i].minP[0]);
    var ymin=parseInt(regiones[i].minP[1]);
    var w=xmax-xmin;var h=ymax-ymin;
    canvas.add(new fabric.Rect({
      left: xmin,
      top: ymin,
      width: w,
      height: h,
      angle: 0,
      evented: false,
      fill: 'transparent',
       strokeWidth: 2,
       strokeDashArray: [10, 5],
       stroke: color
    }));

    var text = new fabric.Text(regiones[i].tag, {
      fontSize: 15,
      evented: false,
      left: xmin,
      top: ymin
    });
    canvas.add(text);
  }
}

function pintarEncontrados(Ids){
  for (var i = 0; i < Ids.length ; i++) {
    polyToRender[parseInt(Ids[i])].setColor(colorFounded)
  }
}

function enlazarEncontrados(obj){
  Ids = obj.data
  x = obj.x
  y = obj.y
  for (var i = 0; i < Ids.length ; i++) {
    polyToRender[parseInt(Ids[i])].setColor(colorFounded)
    console.log(poligonos[parseInt(Ids[i])])
    if(poligonos[parseInt(Ids[i])].length == 2){
      canvas.add(new fabric.Line([x, y, poligonos[parseInt(Ids[i])][0], poligonos[parseInt(Ids[i])][1]], {
        stroke:colorFounded,
        strokeWidth:2,
      }));
    } else if(poligonos[parseInt(Ids[i])].length > 2){
      canvas.add(new fabric.Line([x, y, poligonos[parseInt(Ids[i])][0][0], poligonos[parseInt(Ids[i])][0][1]], {
        stroke:colorFounded,
        strokeWidth:2,
      }));
    }
  }
}

function getRandomColor() {
  var letters = '0123456789ABCDEF';
  var color = '#';
  for (var i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}

function minmax_pol(pol){
  var temp=0;
  var min=[pol[0][0], pol[0][1]], max =[pol[0][0], pol[0][1]];
  //cuudado con maximo por si deciden cambiar dimension de canvas
  for (var j = 1; j< pol.length; j++) {
    if(pol[j][0]<min[0]){
      min[0] = pol[j][0]
    }
    if(pol[j][0]>max[0]){
      max[0] = pol[j][0]
    }
    if(pol[j][1]<min[1]){
      min[1] = pol[j][1]
    }
    if(pol[j][1]>max[1]){
      max[1] = pol[j][1]
    }
  }
  return [min, max]
}

function showKNNOptions() {
  var e = document.getElementById("esc");
  if (knnIsCheck.checked){
    e.style.display = "block";
  } else {
    e.style.display = "none";
  }
}
