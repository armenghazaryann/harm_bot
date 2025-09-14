variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  type        = string
}

variable "release_name" {
  description = "Helm release name"
  type        = string
  default     = "rag-etl"
}

variable "namespace" {
  description = "Kubernetes namespace for the release"
  type        = string
  default     = "default"
}

variable "chart_path" {
  description = "Local path to the Helm chart"
  type        = string
  default     = "../helm"
}

variable "chart_version" {
  description = "Version of the Helm chart to deploy"
  type        = string
  default     = "0.1.0"
}

variable "values_file" {
  description = "Path to the Helm values.yaml file"
  type        = string
  default     = "../helm/values.yaml"
}

variable "image_repository" {
  description = "Docker image repository for the RAG ETL service"
  type        = string
  default     = "your-docker-registry/experiment-rag-etl"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "replica_count" {
  description = "Number of replicas for the deployment"
  type        = number
  default     = 1
}
