import '@xyflow/react/dist/style.css';
import '../../../brand/tokens.css';
import './styles.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import * as core from '@neuroloom/core';
import { App } from './App';
import { nodeFaces } from './canvas/face';
import { useFlow } from './canvas/flowMode';
import { benchEnabled, residualChain } from './lib/bench';
import { useStudio } from './store';

// Lets measurements and debugging reach the editor (development, or `?bench`).
if (benchEnabled) Object.assign(window, { __neuroloom: { useStudio, useFlow, residualChain, core, nodeFaces } });

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
