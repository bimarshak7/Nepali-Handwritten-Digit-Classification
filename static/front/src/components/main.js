import React,{useState, useEffect, useRef} from 'react';
import Canvas from './canvas.js';

const Main = () =>{
	const stageRef = useRef(null);
  const [result,setResult] = useState();
	const handleSubmit = () => {
  	const uri = stageRef.current.toDataURL();
     fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ uri })
  }).then(res=>res.json().then(m=>setResult(m.msg)));
  }
	return(
    <div>
		<Canvas stageRef={stageRef} handleSubmit={handleSubmit}/>
    <h2>Prediction: {result}</h2>
    </div>
		)
}

export default Main;