getDistance = (point, oldrect, distance=1.0) => {
  var inside=false;
  var d;
  // A: [x-distance, x] x [y-height, y]
  if(oldrect.x - distance <= point.x && point.x <=oldrect.x && oldrect.y - oldrect.height <= point.y && point.y <= oldrect.y){
    d = Math.sqrt(Math.pow(oldrect.x - point.x, 2));
  }
  // B: [x, x+width] x [y, y+distance]
  else if(oldrect.x  <= point.x && point.x <= oldrect.x + oldrect.width && oldrect.y <= point.y && point.y <= oldrect.y + distance){
    d = Math.sqrt(Math.pow(point.y - oldrect.y, 2));
  }
  // C: [x, x+width] x [y-height-distance, y-height]
  else if(oldrect.x <= point.x && point.x <=oldrect.x + oldrect.width && oldrect.y -oldrect.height - distance <= point.y && point.y <= oldrect.y - oldrect.height){
    d = Math.sqrt(Math.pow(oldrect.y -oldrect.height- point.y, 2));
  }
  // D: [x+widhth, x+widht+distance] x [y-height, y]
  else if(oldrect.x +oldrect.width <= point.x && point.x <= oldrect.x + oldrect.width + distance && oldrect.y - oldrect.height <= point.y && point.y <= oldrect.y ){
    d = Math.sqrt(Math.pow(oldrect.x + oldrect.width + distance - point.x, 2));
  }
  // E: [x-distance,x] x [y, y+distance]
  else if(oldrect.x - distance <= point.x && point.x <=oldrect.x && oldrect.y <= point.y && point.y <= oldrect.y + distance) {
    d = Math.sqrt(Math.pow(oldrect.x - point.x,2) + Math.pow(oldrect.y - point.y,2))
  }
  // F: [x+width, x+width+distance] x [y, y+distance]
  else if(oldrect.x +oldrect.width <= point.x && point.x <= oldrect.x + oldrect.width + distance && oldrect.y <= point.y && point.y <= oldrect.y + distance){
    d = Math.sqrt(Math.pow(oldrect.x - point.x,2) + Math.pow(oldrect.y - point.y,2))
  }
  // G: [x+width, x+width+distance] x [y-height-distance, y-height]
  else if(oldrect.x +oldrect.width <= point.x && point.x <= oldrect.x + oldrect.width + distance && oldrect.y-oldrect.height-distance <= point.y && point.y <= oldrect.y-oldrect.height) {
    d = Math.sqrt(Math.pow(oldrect.x - point.x,2) + Math.pow(oldrect.y - point.y,2))
  }
  // H: [x -distance, x] x [y-height-distance, y-height]
  else if(oldrect.x - distance <= point.x && point.x <=oldrect.x && oldrect.y-oldrect.height-distance <= point.y && point.y <= oldrect.y-oldrect.height) {
    d = Math.sqrt(Math.pow(oldrect.x - point.x,2) + Math.pow(oldrect.y - point.y,2))
  } else {
    d=1
  }
  return 1/d;
}

getWeights = (data, oldrect) =>{
  var c_W = [];
  data.map(function(obj){
    var point = {
      x:obj[0],
      y:obj[1]
    }
    c_W.push(getDistance(point, oldrect));
  })
  console.log("selected points:", c_W.count(1))
  return c_W
}

Object.defineProperties(Array.prototype, {
    count: {
        value: function(value) {
            return this.filter(x => x==value).length;
        }
    }
});
