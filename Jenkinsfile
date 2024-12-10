pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub')  // DockerHub credentials in Jenkins
        REGISTRY_NAME = 'yourdockerhubusername/music-genre-classification' // Replace with your DockerHub username and repository name
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    userRemoteConfigs: [[url: 'https://github.com/ahmed00078/music-genre-classification']] // Replace with your GitHub repo URL
                ])
            }
        }

        stage('Test if it works') {
            steps {
                script {
                    sh 'echo "Hello, World!"'
                }
            }
        }

        stage('Build SVM Service') {
            steps {
                script {
                    sh '''
                    if [ -f docker-compose.yml ]; then
                        docker-compose build svm-service
                    else
                        echo "docker-compose.yml not found" && exit 1
                    fi
                    '''
                }
            }
        }

        stage('Build VGG19 Service') {
            steps {
                script {
                    sh '''
                    if [ -f docker-compose.yml ]; then
                        docker-compose build vgg19-service
                    else
                        echo "docker-compose.yml not found" && exit 1
                    fi
                    '''
                }
            }
        }

        stage('Build Frontend App') {
            steps {
                script {
                    sh '''
                    if [ -f docker-compose.yml ]; then
                        docker-compose build frontend
                    else
                        echo "docker-compose.yml not found" && exit 1
                    fi
                    '''
                }
            }
        }

        stage('Push Docker Images to DockerHub') {
            steps {
                script {
                    sh '''
                    docker login -u $DOCKERHUB_CREDENTIALS_USR -p $DOCKERHUB_CREDENTIALS_PSW
                    docker-compose push
                    '''
                }
            }
        }

        stage('Deploy to Azure') {
            steps {
                script {
                    // Deploy the containers to Azure Container Instances (or other service like AKS)
                    sh '''
                    # Deploy the Frontend
                    az container create --resource-group MusicGenreGroup --name music-genre-frontend \
                    --image ${REGISTRY_NAME}-frontend \
                    --ports 8501

                    # Deploy the SVM service
                    az container create --resource-group MusicGenreGroup --name music-genre-svm-service \
                    --image ${REGISTRY_NAME}-svm-service \
                    --ports 5001

                    # Deploy the VGG19 service
                    az container create --resource-group MusicGenreGroup --name music-genre-vgg19-service \
                    --image ${REGISTRY_NAME}-vgg19-service \
                    --ports 5002
                    '''
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished execution.'
            script {
                sh '''
                if [ -f docker-compose.yml ]; then
                    docker-compose down || true
                fi
                '''
            }
        }
        success {
            echo 'Pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}