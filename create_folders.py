import os

folders = [
    'disaster_mesh',
    'disaster_mesh/core',
    'disaster_mesh/algorithms',
    'disaster_mesh/baselines',
    'disaster_mesh/environment',
    'disaster_mesh/training',
    'disaster_mesh/evaluation',
    'disaster_mesh/visualization',
    'disaster_mesh/data/crawdad',
    'disaster_mesh/data/synthetic',
    'disaster_mesh/models/saved',
    'disaster_mesh/notebooks'
]

for f in folders:
    os.makedirs(os.path.join(r"c:\Academics\6th Sem\Minor Project\DisasterMesh", f), exist_ok=True)

print("Folders created.")
