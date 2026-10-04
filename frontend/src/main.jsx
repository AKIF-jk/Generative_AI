import React from 'react';
import { createRoot } from 'react-dom/client';
import FaceToSketchGenerator from './workspaces/FaceToSketchGenerator';
import RestorationWorkspace from './workspaces/RestorationWorkspace';
import './styles.css';
import './restoration.css';

function App() {
  const [tab, setTab] = React.useState('task4');
  return <div><header className="app-nav"><a className="brand" href="#top"><span className="brand-icon">✦</span><span>Sketchly</span></a><nav><button className={tab === 'task1' ? 'active' : ''} onClick={() => setTab('task1')}>Task 1 <small>Universal</small></button><button className={tab === 'task2' ? 'active' : ''} onClick={() => setTab('task2')}>Task 2 <small>Hard-routed</small></button><button className={tab === 'task4' ? 'active' : ''} onClick={() => setTab('task4')}>Task 4 <small>Face-to-sketch</small></button></nav></header>{tab === 'task4' ? <FaceToSketchGenerator /> : <main className="sketch-main tab-main"><RestorationWorkspace task={tab === 'task1' ? 1 : 2} /></main>}</div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);
