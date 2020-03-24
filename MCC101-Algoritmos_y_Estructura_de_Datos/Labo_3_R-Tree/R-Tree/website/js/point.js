prototypefabric.point = {
    points: [],
    drawPoint : function() {
        pointMode = true;
        pointArray = new Array();
        polygonMode = false;
        lineArray = new Array();
        activeLine;
    },
    addPoint : function(options) {
        var random = Math.floor(Math.random() * (max - min + 1)) + min;
        var id = new Date().getTime() + random;
        var circle = new fabric.Circle({
            radius: 5,
            fill: 'red',
            stroke: '#333333',
            strokeWidth: 0.5,
            left: (options.e.layerX/canvas.getZoom()),
            top: (options.e.layerY/canvas.getZoom()),
            selectable: false,
            hasBorders: false,
            hasControls: false,
            originX:'center',
            originY:'center',
            id:id
        });
        if(pointArray.length == 0){
            circle.set({
                fill:'red'
            })
        }
        var points = [(options.e.layerX/canvas.getZoom()),(options.e.layerY/canvas.getZoom()),(options.e.layerX/canvas.getZoom()),(options.e.layerY/canvas.getZoom())];
        line = new fabric.Line(points, {
            strokeWidth: 2,
            fill: '#999999',
            stroke: '#999999',
            class:'line',
            originX:'center',
            originY:'center',
            selectable: false,
            hasBorders: false,
            hasControls: false,
            evented: false
        });
        setText(points[0],points[1]);
        pointArray.push(circle);
        canvas.add(circle);
        polyToRender[polCount] = circle
        pointMode = false;
       // alert("hola desde addpoint 2");
    },

     limpiarpoint : function() {
        circle.length=0;
      //  alert("limpiar punto");
    }
};


function setText(points1,points2) {
    prototypefabric.point.points = [parseInt(points1), parseInt(points2)]
};
