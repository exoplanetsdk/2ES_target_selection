import os

def create_project_structure():
    # Define the base directory structure
    structure = {
        'src': {
            '__init__.py': '',
            'main.py': '',
            'config.py': '',
            'data_processing.py': '',
            'gaia_queries.py': '',
            'plotting.py': '',
            'stellar_calculations.py': '',
            'simbad_integration.py': '',
            'utils.py': ''
        },
        'data': {},
        'results': {}
    }

    # Create base project directory
    project_dir = 'project'
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # Create the directory structure and files
    for dir_name, contents in structure.items():
        # Create directory
        dir_path = os.path.join(project_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Create files in the directory
        for file_name, content in contents.items():
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(content)

    print("Project structure created successfully!")

# Create the project structure
create_project_structure()
