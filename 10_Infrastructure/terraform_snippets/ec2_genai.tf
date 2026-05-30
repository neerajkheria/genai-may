# Terraform snippet — EC2 instance for GenAI training
# provider "aws" { region = "us-east-1" }

variable "instance_type" { default = "t3.large" }
variable "key_name"      { description = "EC2 key pair name" }

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "genai_sg" {
  name        = "genai-training-sg"
  description = "Security group for GenAI training EC2"

  ingress { from_port = 22   to_port = 22   protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  ingress { from_port = 8000 to_port = 8005 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  ingress { from_port = 8443 to_port = 8443 protocol = "tcp" cidr_blocks = ["0.0.0.0/0"] }
  egress  { from_port = 0   to_port = 0   protocol = "-1" cidr_blocks = ["0.0.0.0/0"] }
}

resource "aws_instance" "genai_ec2" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  key_name      = var.key_name
  vpc_security_group_ids = [aws_security_group.genai_sg.id]

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  user_data = file("${path.module}/../aws_ec2_setup/setup.sh")

  tags = { Name = "genai-training-instance", Project = "enterprise-ai-bootcamp" }
}

output "public_ip"  { value = aws_instance.genai_ec2.public_ip }
output "ssh_command" { value = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_instance.genai_ec2.public_ip}" }
