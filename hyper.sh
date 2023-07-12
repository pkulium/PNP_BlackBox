#!/bin/bash

# Run your commands
python optimize_mask_cifar.py --anp-eps 0.1 >> eps.txt || echo "command1 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.2 >> eps.txt || echo "command2 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.3 >> eps.txt || echo "command3 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.4 >> eps.txt || echo "command4 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.5 >> eps.txt || echo "command5 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.6 >> eps.txt || echo "command6 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.7 >> eps.txt || echo "command7 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.8 >> eps.txt || echo "command8 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 0.9 >> eps.txt || echo "command9 failed" >> eps.txt
python optimize_mask_cifar.py --anp-eps 1.0 >> eps.txt || echo "command10 failed" >> eps.txt

# Run your commands
python optimize_mask_cifar.py --anp-alpha 0.1 >> alpha.txt || echo "command1 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.2 >> alpha.txt || echo "command2 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.3 >> alpha.txt || echo "command3 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.4 >> alpha.txt || echo "command4 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.5 >> alpha.txt || echo "command5 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.6 >> alpha.txt || echo "command6 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.7 >> alpha.txt || echo "command7 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.8 >> alpha.txt || echo "command8 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 0.9 >> alpha.txt || echo "command9 failed" >> alpha.txt
python optimize_mask_cifar.py --anp-alpha 1.0 >> alpha.txt || echo "command10 failed" >> alpha.txt
