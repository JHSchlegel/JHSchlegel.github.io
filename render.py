import os
import subprocess

subprocess.run(['quarto', 'render', 'posts/03-04-2025_intro_to_sde/index.qmd', '--to', 'html'])
