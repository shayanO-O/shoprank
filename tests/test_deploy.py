from pathlib import Path
import pytest
import yaml

def test_dockerfile_syntax_and_structure():
    df_path = Path("Dockerfile")
    assert df_path.exists()
    content = df_path.read_text(encoding="utf-8")
    
    assert "FROM python:3.11-slim as builder" in content
    assert "FROM python:3.11-slim as runner" in content
    assert "useradd" in content and "appuser" in content
    assert "EXPOSE 8000 8080" in content
    assert "HEALTHCHECK" in content
    assert "uvicorn" in content

def test_docker_compose_syntax_and_services():
    compose_path = Path("deploy/docker-compose.yml")
    assert compose_path.exists()
    
    with open(compose_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert "services" in data
    services = data["services"]
    
    expected_services = ["redis", "champion", "challenger", "router", "prometheus", "grafana"]
    for s in expected_services:
        assert s in services, f"Missing service {s} in docker-compose.yml"
        
    # Verify ports
    assert "8000:8000" in services["champion"]["ports"]
    assert "8001:8001" in services["challenger"]["ports"]
    assert "8080:8080" in services["router"]["ports"]
    assert "6379:6379" in services["redis"]["ports"]
    assert "9090:9090" in services["prometheus"]["ports"]
    assert "3000:3000" in services["grafana"]["ports"]

def test_prometheus_config():
    prom_path = Path("deploy/prometheus/prometheus.yml")
    assert prom_path.exists()
    
    with open(prom_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert "scrape_configs" in data
    job_names = [j["job_name"] for j in data["scrape_configs"]]
    assert "shoprank-router" in job_names
    assert "shoprank-champion" in job_names
    assert "shoprank-challenger" in job_names

def test_kubernetes_manifests():
    k8s_dir = Path("deploy/k8s")
    assert k8s_dir.exists()
    manifests = list(k8s_dir.glob("*.yaml"))
    assert len(manifests) >= 4
    
    for mf in manifests:
        with open(mf, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            for doc in docs:
                if not doc:
                    continue
                assert "apiVersion" in doc, f"Missing apiVersion in {mf}"
                assert "kind" in doc, f"Missing kind in {mf}"
                assert "metadata" in doc, f"Missing metadata in {mf}"
                assert "spec" in doc, f"Missing spec in {mf}"

    # Verify Gateway API HTTPRoute exists
    route_file = k8s_dir / "gateway-canary.yaml"
    assert route_file.exists()
    with open(route_file, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert doc["kind"] == "HTTPRoute"
    assert doc["apiVersion"] == "gateway.networking.k8s.io/v1"

def test_github_actions_workflow():
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists()
    
    with open(ci_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert "name" in data
    assert "jobs" in data
    jobs = data["jobs"]
    assert "test" in jobs
    assert "validate-deployment" in jobs
    assert "build-and-push" in jobs
    assert "smoke-test" in jobs
