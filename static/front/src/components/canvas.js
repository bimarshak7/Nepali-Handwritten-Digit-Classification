import React,{useState, useEffect, useRef} from 'react';
import { Stage, Layer, Line, Rect } from 'react-konva';

const Canvas = ({stageRef, handleSubmit}) => {
	const [tool, setTool] = useState('pen');
  	const [lines, setLines] = useState([]);
  	const isDrawing = useRef(false);
  	  const handleMouseDown = (e) => {
    isDrawing.current = true;
    const pos = e.target.getStage().getPointerPosition();
    setLines([...lines, { tool, points: [pos.x, pos.y] }]);
  };

  const handleMouseMove = (e) => {
    // no drawing - skipping
    if (!isDrawing.current) {
      return;
    }
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();
    let lastLine = lines[lines.length - 1];
    // add point
    lastLine.points = lastLine.points.concat([point.x, point.y]);

    // replace last
    lines.splice(lines.length - 1, 1, lastLine);
    setLines(lines.concat());
  };

  const handleMouseUp = () => {
    isDrawing.current = false;
  };
  
  const handleClear = (e) => setLines([]);
  const MAIN_CONTAINER_STYLE = {
  width: "400px",
  height: "400px",
  margin: "0 auto",
	};
  const SKETCH_CONTAINER_STYLE = {
  border: "1px solid green",
  backgroundColor: "white",
  width: "300px",
  height: "300px",
  margin: "auto"
	};
	return(<div style={MAIN_CONTAINER_STYLE}>
		<select
        value={tool}
        onChange={(e) => {
          setTool(e.target.value);
        }}
      >
        <option value="pen">Pen</option>
        <option value="eraser">Eraser</option>
      </select>
		<div style={SKETCH_CONTAINER_STYLE}>
		<Stage
        width={300}
        height={300}
        ref={stageRef}
        onMouseDown={handleMouseDown}
        onMousemove={handleMouseMove}
        onMouseup={handleMouseUp}
      >
      <Layer>
          <Rect
          x={0}
          y={0}
          width={300}
          height={300}
          fill="white"
        />
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line.points}
              stroke="#000"
              strokeWidth={10}
              tension={0.6}
              lineCap="round"
              globalCompositeOperation={
                line.tool === 'eraser' ? 'destination-out' : 'source-over'
              }
            />
          ))}
        </Layer>
      	</Stage>
		</div>
      	<button onClick={handleClear}>Clear</button>
      	<button onClick={handleSubmit}>Predict</button>
		</div>
		)
}

export default Canvas;