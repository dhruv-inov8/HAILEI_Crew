"""
Test Suite: Phase 4.3 - Docker Containerization

Tests for Docker configuration, deployment scripts, and container functionality.
"""

import unittest
import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDockerConfiguration(unittest.TestCase):
    """Test Docker configuration files"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path(__file__).parent.parent
        self.docker_files = {
            'Dockerfile': self.project_root / 'Dockerfile',
            'docker-compose.yml': self.project_root / 'docker-compose.yml',
            '.dockerignore': self.project_root / '.dockerignore',
            'deploy.sh': self.project_root / 'deploy.sh'
        }
        self.config_files = {
            'nginx.conf': self.project_root / 'config' / 'nginx.conf',
            'redis.conf': self.project_root / 'config' / 'redis.conf',
            'init.sql': self.project_root / 'config' / 'init.sql',
            'prometheus.yml': self.project_root / 'config' / 'prometheus.yml'
        }
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and has correct structure"""
        dockerfile = self.docker_files['Dockerfile']
        self.assertTrue(dockerfile.exists(), "Dockerfile should exist")
        
        content = dockerfile.read_text()
        
        # Check essential Dockerfile components
        self.assertIn('FROM python:3.10-slim', content)
        self.assertIn('WORKDIR /app', content)
        self.assertIn('COPY requirements', content)
        self.assertIn('pip install', content)
        self.assertIn('EXPOSE', content)
        self.assertIn('CMD', content)
        
        # Check security practices
        self.assertIn('adduser', content)  # Non-root user
        self.assertIn('USER hailei', content)  # Switch to non-root
        
        # Check health check
        self.assertIn('HEALTHCHECK', content)
    
    def test_docker_compose_structure(self):
        """Test docker-compose.yml structure"""
        compose_file = self.docker_files['docker-compose.yml']
        self.assertTrue(compose_file.exists(), "docker-compose.yml should exist")
        
        content = compose_file.read_text()
        
        # Check version
        self.assertIn('version:', content)
        
        # Check essential services
        required_services = ['hailei-api', 'redis', 'nginx', 'postgres']
        for service in required_services:
            self.assertIn(service, content)
        
        # Check networking
        self.assertIn('networks:', content)
        self.assertIn('hailei-network', content)
        
        # Check volumes
        self.assertIn('volumes:', content)
        
        # Check health checks
        self.assertIn('healthcheck:', content)
    
    def test_dockerignore_file(self):
        """Test .dockerignore file"""
        dockerignore = self.docker_files['.dockerignore']
        self.assertTrue(dockerignore.exists(), ".dockerignore should exist")
        
        content = dockerignore.read_text()
        
        # Check essential exclusions
        exclusions = [
            '__pycache__',
            '.git',
            'tests/',
            'logs/',
            '.env',
            'node_modules',
            'Dockerfile',
            'docker-compose'
        ]
        
        for exclusion in exclusions:
            self.assertIn(exclusion, content)
    
    def test_deployment_script(self):
        """Test deployment script"""
        deploy_script = self.docker_files['deploy.sh']
        self.assertTrue(deploy_script.exists(), "deploy.sh should exist")
        
        # Check if script is executable
        self.assertTrue(os.access(deploy_script, os.X_OK), "deploy.sh should be executable")
        
        content = deploy_script.read_text()
        
        # Check script structure
        self.assertIn('#!/bin/bash', content)
        self.assertIn('set -e', content)  # Exit on error
        
        # Check essential functions
        functions = [
            'check_prerequisites',
            'deploy',
            'health_check',
            'backup_data',
            'cleanup'
        ]
        
        for function in functions:
            self.assertIn(function, content)
    
    def test_config_files_exist(self):
        """Test that all configuration files exist"""
        for name, path in self.config_files.items():
            with self.subTest(config_file=name):
                self.assertTrue(path.exists(), f"{name} should exist")
    
    def test_nginx_config(self):
        """Test nginx configuration"""
        nginx_config = self.config_files['nginx.conf']
        content = nginx_config.read_text()
        
        # Check essential nginx directives
        self.assertIn('upstream hailei_backend', content)
        self.assertIn('proxy_pass', content)
        self.assertIn('location /api/', content)
        self.assertIn('location /ws/', content)
        self.assertIn('location /frontend/', content)
        
        # Check security headers
        self.assertIn('X-Frame-Options', content)
        self.assertIn('X-XSS-Protection', content)
        
        # Check gzip compression
        self.assertIn('gzip on', content)
        
        # Check rate limiting
        self.assertIn('limit_req_zone', content)
    
    def test_redis_config(self):
        """Test Redis configuration"""
        redis_config = self.config_files['redis.conf']
        content = redis_config.read_text()
        
        # Check essential Redis settings
        self.assertIn('bind 0.0.0.0', content)
        self.assertIn('port 6379', content)
        self.assertIn('appendonly yes', content)
        self.assertIn('maxmemory', content)
        self.assertIn('maxmemory-policy', content)
    
    def test_postgres_init_script(self):
        """Test PostgreSQL initialization script"""
        init_script = self.config_files['init.sql']
        content = init_script.read_text()
        
        # Check schema creation
        self.assertIn('CREATE SCHEMA', content)
        self.assertIn('sessions', content)
        self.assertIn('metrics', content)
        
        # Check table creation
        tables = [
            'conversation_sessions',
            'conversation_context',
            'agent_executions',
            'user_feedback'
        ]
        
        for table in tables:
            self.assertIn(table, content)
        
        # Check indexes
        self.assertIn('CREATE INDEX', content)
    
    def test_prometheus_config(self):
        """Test Prometheus configuration"""
        prometheus_config = self.config_files['prometheus.yml']
        content = prometheus_config.read_text()
        
        # Check global settings
        self.assertIn('global:', content)
        self.assertIn('scrape_interval:', content)
        
        # Check job configurations
        jobs = ['hailei-api', 'redis', 'postgres', 'nginx']
        for job in jobs:
            self.assertIn(job, content)
    
    def test_environment_template(self):
        """Test environment template file"""
        env_example = self.project_root / '.env.example'
        self.assertTrue(env_example.exists(), ".env.example should exist")
        
        content = env_example.read_text()
        
        # Check essential environment variables
        env_vars = [
            'ENVIRONMENT',
            'JWT_SECRET_KEY',
            'POSTGRES_PASSWORD',
            'REDIS_HOST',
            'API_PORT'
        ]
        
        for var in env_vars:
            self.assertIn(var, content)


class TestDockerBuild(unittest.TestCase):
    """Test Docker build process"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path(__file__).parent.parent
        self.image_name = "hailei:test"
    
    def test_docker_build(self):
        """Test that Docker image builds successfully"""
        # Skip if Docker is not available
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("Docker not available")
        
        # Build the image
        build_cmd = [
            'docker', 'build',
            '-t', self.image_name,
            '-f', str(self.project_root / 'Dockerfile'),
            str(self.project_root)
        ]
        
        try:
            result = subprocess.run(
                build_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            self.assertEqual(result.returncode, 0, "Docker build should succeed")
            
            # Verify image exists
            inspect_cmd = ['docker', 'inspect', self.image_name]
            result = subprocess.run(inspect_cmd, check=True, capture_output=True)
            self.assertEqual(result.returncode, 0, "Built image should be inspectable")
            
        except subprocess.TimeoutExpired:
            self.fail("Docker build timed out")
        except subprocess.CalledProcessError as e:
            self.fail(f"Docker build failed: {e.stderr}")
        
        # Clean up
        try:
            subprocess.run(['docker', 'rmi', self.image_name], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass  # Cleanup failure is not critical
    
    def test_docker_image_security(self):
        """Test Docker image security practices"""
        # Skip if Docker is not available
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("Docker not available")
        
        # Build image first
        build_cmd = [
            'docker', 'build',
            '-t', self.image_name,
            '-f', str(self.project_root / 'Dockerfile'),
            str(self.project_root)
        ]
        
        try:
            subprocess.run(build_cmd, check=True, capture_output=True, timeout=300)
            
            # Check that container runs as non-root user
            run_cmd = [
                'docker', 'run', '--rm',
                self.image_name,
                'whoami'
            ]
            
            result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
            self.assertNotEqual(result.stdout.strip(), 'root', "Container should not run as root")
            
        except subprocess.CalledProcessError as e:
            self.fail(f"Security test failed: {e}")
        except subprocess.TimeoutExpired:
            self.fail("Docker build timed out")
        
        # Clean up
        try:
            subprocess.run(['docker', 'rmi', self.image_name], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass


class TestDeploymentScripts(unittest.TestCase):
    """Test deployment script functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path(__file__).parent.parent
        self.deploy_script = self.project_root / 'deploy.sh'
    
    def test_deploy_script_syntax(self):
        """Test deployment script syntax"""
        # Check bash syntax
        try:
            result = subprocess.run(
                ['bash', '-n', str(self.deploy_script)],
                check=True,
                capture_output=True,
                text=True
            )
            self.assertEqual(result.returncode, 0, "Deploy script should have valid bash syntax")
        except subprocess.CalledProcessError as e:
            self.fail(f"Deploy script syntax error: {e.stderr}")
    
    def test_deploy_script_help(self):
        """Test deployment script help output"""
        try:
            result = subprocess.run(
                [str(self.deploy_script)],
                capture_output=True,
                text=True
            )
            
            # Should show usage information
            self.assertIn('Usage:', result.stdout)
            self.assertIn('deploy', result.stdout)
            self.assertIn('start', result.stdout)
            self.assertIn('stop', result.stdout)
            
        except subprocess.CalledProcessError as e:
            self.fail(f"Deploy script help failed: {e}")


class TestProductionReadiness(unittest.TestCase):
    """Test production readiness aspects"""
    
    def test_security_configurations(self):
        """Test security configurations"""
        project_root = Path(__file__).parent.parent
        
        # Check that sensitive files are in .dockerignore
        dockerignore = project_root / '.dockerignore'
        content = dockerignore.read_text()
        
        sensitive_patterns = ['.env', '*.key', '*.pem', 'secrets']
        for pattern in sensitive_patterns:
            self.assertIn(pattern, content, f"Sensitive pattern {pattern} should be in .dockerignore")
    
    def test_health_check_endpoints(self):
        """Test that health check configurations are present"""
        project_root = Path(__file__).parent.parent
        
        # Check Dockerfile health check
        dockerfile = project_root / 'Dockerfile'
        dockerfile_content = dockerfile.read_text()
        self.assertIn('HEALTHCHECK', dockerfile_content)
        
        # Check docker-compose health checks
        compose_file = project_root / 'docker-compose.yml'
        compose_content = compose_file.read_text()
        self.assertIn('healthcheck:', compose_content)
    
    def test_resource_limits(self):
        """Test that resource limits are configured"""
        project_root = Path(__file__).parent.parent
        
        # Check docker-compose for resource limits
        compose_file = project_root / 'docker-compose.yml'
        compose_content = compose_file.read_text()
        
        # Should have restart policies
        self.assertIn('restart:', compose_content)
        
        # Check Redis memory limit
        redis_config = project_root / 'config' / 'redis.conf'
        redis_content = redis_config.read_text()
        self.assertIn('maxmemory', redis_content)
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        project_root = Path(__file__).parent.parent
        
        # Check that log directories are created
        compose_file = project_root / 'docker-compose.yml'
        compose_content = compose_file.read_text()
        self.assertIn('logs:', compose_content)
        
        # Check nginx logging
        nginx_config = project_root / 'config' / 'nginx.conf'
        nginx_content = nginx_config.read_text()
        self.assertIn('access_log', nginx_content)
        self.assertIn('error_log', nginx_content)
    
    def test_monitoring_setup(self):
        """Test monitoring and metrics setup"""
        project_root = Path(__file__).parent.parent
        
        # Check Prometheus configuration
        prometheus_config = project_root / 'config' / 'prometheus.yml'
        self.assertTrue(prometheus_config.exists())
        
        # Check docker-compose includes monitoring services
        compose_file = project_root / 'docker-compose.yml'
        compose_content = compose_file.read_text()
        self.assertIn('prometheus:', compose_content)
        self.assertIn('grafana:', compose_content)


def run_phase_4_3_tests():
    """Run all Phase 4.3 tests and return results"""
    print("🧪 Running Phase 4.3 Tests: Docker Containerization")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDockerConfiguration,
        TestDockerBuild,
        TestDeploymentScripts,
        TestProductionReadiness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print(f"\\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    if hasattr(result, 'skipped') and result.skipped:
        print(f"\\n⏭️  Skipped:")
        for test, reason in result.skipped:
            print(f"   - {test}: {reason}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    run_phase_4_3_tests()