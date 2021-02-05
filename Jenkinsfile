pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh 'sudo apt-get install -y python3-venv build-essential libblas-dev liblapack-dev python3-tk python3-dev'
                withPythonEnv('CPython-3.6') {
                        sh '''
                           pip3 list
                           pip3 install -r requirements.txt
                           python3 setup.py develop
                           export MPLBACKEND="agg"
                           pytest -v
                        '''
                }
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying.'
            }
        }
    }
}
