import { SCHEMA_VERSION, auto, literal, type LoomProject } from '@neuroloom/core';

/** The project shown on first launch: a small MLP for MNIST-sized inputs. */
export function sampleProject(): LoomProject {
  return {
    schemaVersion: SCHEMA_VERSION,
    name: 'MLP',
    entry: 'main',
    modules: {
      main: {
        id: 'main',
        className: 'MLP',
        nodes: {
          n_x: { id: 'n_x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: literal('B, 784') } },
          n_fc1: {
            id: 'n_fc1',
            kind: 'layer',
            type: 'torch.nn.Linear',
            name: 'fc1',
            params: { in_features: auto(), out_features: literal(256) },
          },
          n_act1: { id: 'n_act1', kind: 'layer', type: 'torch.nn.ReLU', name: 'act1', params: {} },
          n_fc2: {
            id: 'n_fc2',
            kind: 'layer',
            type: 'torch.nn.Linear',
            name: 'fc2',
            params: { in_features: auto(), out_features: literal(10) },
          },
          n_out: { id: 'n_out', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} },
        },
        edges: [
          { id: 'e1', source: { node: 'n_x', port: 'out' }, target: { node: 'n_fc1', port: 'in' } },
          { id: 'e2', source: { node: 'n_fc1', port: 'out' }, target: { node: 'n_act1', port: 'in' } },
          { id: 'e3', source: { node: 'n_act1', port: 'out' }, target: { node: 'n_fc2', port: 'in' } },
          { id: 'e4', source: { node: 'n_fc2', port: 'out' }, target: { node: 'n_out', port: 'in' } },
        ],
        layout: {
          n_x: { x: 0, y: 40 },
          n_fc1: { x: 260, y: 40 },
          n_act1: { x: 520, y: 40 },
          n_fc2: { x: 780, y: 40 },
          n_out: { x: 1040, y: 40 },
        },
      },
    },
  };
}

export const SAMPLE_IMPORT_CODE = `import torch
import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 32)
        self.dec1 = nn.Linear(32, 256)
        self.dec2 = nn.Linear(256, 784)

    def forward(self, x):
        z = F.relu(self.enc1(x))
        z = self.enc2(z)
        h = F.relu(self.dec1(z))
        return self.dec2(h), z
`;
