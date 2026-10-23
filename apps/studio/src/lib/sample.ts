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
        hyperparams: [],
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
        interface: { inputs: ['n_x'], outputs: ['n_out'] },
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

/** Shown in the import dialog: submodules, hyperparameters and a repeat. */
export const SAMPLE_IMPORT_CODE = `import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, hidden_ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * hidden_ratio)
        self.fc2 = nn.Linear(dim * hidden_ratio, dim)

    def forward(self, x):
        h = self.fc2(F.gelu(self.fc1(self.norm(x))))
        return x + h


class Classifier(nn.Module):
    def __init__(self, dim=64, depth=4, num_classes=10):
        super().__init__()
        self.embed = nn.Linear(784, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(torch.flatten(x, 1))
        for block in self.blocks:
            x = block(x)
        return self.head(x)
`;

/** A new document: one input and one output, ready to be wired up. */
export function blankProject(): LoomProject {
  return {
    schemaVersion: SCHEMA_VERSION,
    name: 'Untitled',
    entry: 'main',
    modules: {
      main: {
        id: 'main',
        className: 'Model',
        hyperparams: [],
        nodes: {
          n_x: { id: 'n_x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: literal('B, 784') } },
          n_out: { id: 'n_out', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} },
        },
        edges: [],
        interface: { inputs: ['n_x'], outputs: ['n_out'] },
        layout: { n_x: { x: 0, y: 40 }, n_out: { x: 520, y: 40 } },
      },
    },
  };
}
