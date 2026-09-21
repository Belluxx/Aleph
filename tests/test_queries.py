"""Agent workflows tested at the CLI boundary with offline provider responses."""

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import aleph
from src import capture, common, routes
from src.osm import Source
from tests.fixtures import image_bytes, metadata


def encoded(value):
    return json.dumps(value).encode()


def place(identity=1, name="Test Street", lat=0, lon=0):
    return dict(type="Feature", properties=dict(osm_type="W", osm_id=identity, name=name,
                osm_key="highway", osm_value="residential", city="Test City"),
                geometry=dict(type="Point", coordinates=[lon, lat]))


def way(identity, nodes, points, name="Test Street"):
    return dict(type="way", id=identity, nodes=nodes,
                tags=dict(highway="residential", name=name),
                points=points)


def coverage(views):
    return encoded([None, [None, [
        [[[2, identity], None, [[None, None, lat, lon]]]] for identity, lat, lon in views
    ]]])


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.get = self.enterContext(patch.object(common.Client, "get"))

    def invoke(self, *args):
        with redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()):
            status = aleph.main([*args, "--json", "--cache-dir", str(self.directory / "cache")])
        return status, json.loads(output.getvalue())

    def test_search_reverse_and_nearby_preserve_location_meaning(self):
        self.get.return_value = encoded(dict(features=[place(lat=1, lon=2)]))
        status, result = self.invoke("resolve", "Test Street")
        self.assertEqual(status, 0)
        self.assertEqual(result["results"][0]["id"], "way/1")
        self.assertEqual((result["results"][0]["lat"], result["results"][0]["lon"]), (1, 2))
        status, result = self.invoke("resolve", "--at", "1", "2")
        self.assertEqual(result["mode"], "reverse")
        self.assertEqual(result["results"][0]["distance_m"], 0)
        address = self.get.call_args.args[0]
        self.assertEqual(parse_qs(urlsplit(address).query)["limit"], ["1"])
        data = [
            dict(type="node", id=2, center=(1, 2.0005), tags=dict(name="Cafe", amenity="cafe")),
            dict(type="way", id=3, center=(1, 2.0001), tags=dict(name="Museum", tourism="museum")),
            dict(type="node", id=4, center=(1, 2.1), tags=dict(name="Far away", shop="books")),
        ]
        with patch.object(Source, "data", return_value=(data, dict(osm_data_at="2026-09-20T20:00:00Z"))):
            status, result = self.invoke("resolve", "--at", "1", "2", "--nearby", "--radius", "100")
        self.assertEqual(status, 0)
        self.assertEqual([item["id"] for item in result["results"]], ["way/3", "node/2"])
        self.assertEqual(result["results"][0]["location_type"], "bbox_center")

    def test_ambiguous_name_returns_candidates_without_downloading(self):
        self.get.return_value = encoded(dict(features=[place(), place(2, "Another Street")]))
        status, result = self.invoke("streetview", "--place", "Test", "-o", str(self.directory / "output"))
        self.assertEqual(status, 1)
        self.assertEqual(result["error"]["code"], "ambiguous_place")
        self.assertEqual(len(result["error"]["candidates"]), 2)
        self.assertEqual(self.get.call_count, 1)
        self.assertFalse((self.directory / "output").exists())

    def test_exact_satellite_tile_is_cached_and_can_be_exported_offline(self):
        self.get.return_value = image_bytes((256, 256), "red")
        for attempt in range(2):
            status, result = self.invoke("satellite", "--tile", "19/280337/194891", "-o", str(self.directory))
            self.assertEqual(status, 0)
            self.assertEqual((result["width"], result["height"]), (256, 256))
            self.assertEqual(result["cache"]["hits"], attempt)
            folder = Path(result["folder"])
            run = capture.load(folder)
            capture.export(run, folder, lambda *args: None)
            self.assertTrue(Path(result["path"]).is_file())
        self.assertEqual(self.get.call_count, 1)

    def test_satellite_exceeds_former_size_tile_and_pixel_caps(self):
        self.get.return_value = image_bytes((256, 256), "red")
        status, result = self.invoke("satellite", "--at", "0", "0", "--size", "200001",
                                     "--zoom", "12", "-o", str(self.directory))
        self.assertEqual(status, 0, result)
        self.assertGreater(result["width"] * result["height"], 16_000_000)
        self.assertGreater(self.get.call_count, 128)
        self.assertTrue(Path(result["path"]).is_file())

    def test_single_streetview_uses_actual_camera_and_look_at_without_osm(self):
        def response(address, **kwargs):
            if "/ac/v1" in address:
                return coverage([("panorama_0001", 0, 0.0001), ("panorama_0002", 0, 0.01)])
            if "/photometa/v1" in address:
                return metadata(lon=0.0001)
            if "/thumbnail" in address:
                values = parse_qs(urlsplit(address).query)
                self.assertAlmostEqual(float(values["yaw"][0]), 90, places=3)
                self.assertEqual(values["pitch"], ["10.0"])
                return image_bytes((1024, 576), "blue")
            self.fail(f"Unexpected request: {address}")

        self.get.side_effect = response
        status, result = self.invoke("streetview", "--at", "0", "0", "--look-at", "0", "1",
                                     "--pitch", "10", "-o", str(self.directory))
        self.assertEqual(status, 0)
        photo = result["photos"][0]
        self.assertEqual(photo["pano_id"], "panorama_0001")
        self.assertEqual(photo["lon"], 0.0001)
        self.assertAlmostEqual(photo["distance_m"], 11.12, places=2)
        self.assertEqual(photo["imagery_date"], "2024-06")
        self.assertTrue(Path(photo["path"]).is_file())

    def test_street_sequence_orders_split_ways_and_saves_both_sides(self):
        ways = [way(1, [2, 1], [(0, 0.001), (0, 0)]),
                way(2, [2, 3], [(0, 0.001), (0, 0.002)])]
        self.enterContext(patch.object(Source, "street", return_value=("Test Street", ways)))
        self.enterContext(patch.object(Source, "data", return_value=(ways, {})))
        failure, images = None, 0

        def response(address, **kwargs):
            nonlocal images
            if "/api/?" in address:
                return encoded(dict(features=[place()]))
            if "/ac/v1" in address:
                return coverage([(f"panorama_{i:04d}", 0, lon) for i, lon in enumerate((0, 0.001, 0.002), 1)])
            if "/photometa/v1" in address:
                identity = next(f"panorama_{i:04d}" for i in range(1, 4) if f"panorama_{i:04d}" in address)
                return metadata(identity, lon=(int(identity[-4:]) - 1) * 0.001)
            images += 1
            if failure and images == 2:
                raise failure
            return image_bytes((1024, 576), "green")

        self.get.side_effect = response
        status, result = self.invoke("streetview", "--street", "Test Street", "--stops", "3", "--view", "both", "-o", str(self.directory))
        self.assertEqual(status, 0, result)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["saved_stops"], 3)
        self.assertEqual([p["pano_id"] for p in result["photos"]],
                         [f"panorama_{i:04d}" for i in (1, 1, 2, 2, 3, 3)])
        self.assertEqual([round(p["heading"]) for p in result["photos"]], [0, 180] * 3)
        self.assertAlmostEqual(result["route"]["length_m"], 222.39, places=2)
        status, result = self.invoke("streetview", "--street", "Test Street", "--stops", "101",
                                     "-o", str(self.directory))
        self.assertEqual(status, 0, result)
        self.assertEqual(result["requested_stops"], 101)
        self.assertEqual(result["saved_stops"], 3)
        self.assertEqual(len(result["gaps"]), 98)
        for failure, expected_status in ((OSError("connection lost"), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(failure=type(failure).__name__):
                images = 0
                status, result = self.invoke("streetview", "--street", "Test Street", "--stops", "3",
                                             "--view", "both", "--refresh", "-o", str(self.directory))
                self.assertEqual(status, expected_status, result)
                saved = json.loads((Path(result["error"]["folder"]) / "result.json").read_text())
                self.assertEqual(saved["status"], "partial")
                self.assertEqual(len(saved["photos"]), 1)
                self.assertTrue(Path(saved["photos"][0]["path"]).is_file())

    def test_no_coverage_is_a_structured_error_and_does_not_create_output(self):
        self.get.return_value = coverage([])
        status, result = self.invoke("streetview", "--at", "0", "0", "--radius", "2000",
                                     "-o", str(self.directory / "output"))
        self.assertEqual(status, 1)
        self.assertEqual(result["error"]["code"], "no_coverage")
        self.assertGreater(self.get.call_count, 128)
        self.assertFalse((self.directory / "output").exists())

    def test_invalid_or_conflicting_options_fail_before_requests(self):
        cases = [
            ["resolve", "Rome", "--at", "1", "2"],
            ["resolve", "--at", "nan", "0"],
            ["resolve", "Rome", "--nearby"],
            ["resolve", "--at", "1", "2", "--radius", "20"],
            ["resolve", "--at", "1", "2", "--nearby", "--radius", "5001"],
            ["resolve", "Test", "--limit", "51"],
            ["resolve", "--at", "1", "2", "--nearby", "--limit", "51"],
            ["streetview", "--at", "0", "0", "--place", "Rome"],
            ["streetview", "--at", "0", "0", "--stops", "10"],
            ["streetview", "--street", "Test", "--heading", "90"],
            ["streetview", "--street", "Test", "--stops", "0"],
            ["streetview", "--at", "0", "0", "--radius", "0"],
            ["streetview", "--at", "0", "0", "--radius", "inf"],
            ["satellite", "--at", "0", "0", "--size", "0"],
            ["satellite", "--at", "0", "0", "--size", "nan"],
            ["satellite", "--tile", "19/1"],
            ["satellite", "--tile", "1/2/0"],
            ["satellite", "--tile", "19/1/1", "--zoom", "20"],
            ["satellite", "--bbox", "0", "0", "1", "1", "--size", "20"],
        ]
        for args in cases:
            with self.subTest(args=args):
                status, result = self.invoke(*args)
                self.assertEqual(status, 2, result)
                self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.get.assert_not_called()


class RouteTests(unittest.TestCase):
    def test_branching_streets_return_choices_and_crossings_do_not_connect(self):
        ways = [way(1, [1, 2, 3], [(0, 0), (0, 0.001), (0, 0.002)]),
                way(2, [2, 4], [(0, 0.001), (0.001, 0.001)])]
        self.assertEqual(len(routes.chains(ways)), 3)
        client = Mock()
        client.maps.street.return_value = ("Test Street", ways)
        with self.assertRaises(common.RequestError) as raised:
            routes.resolve(client, dict(id="way/1"))
        self.assertEqual(raised.exception.code, "ambiguous_route")
        choices = raised.exception.details["candidates"]
        selected = routes.resolve(client, dict(id="way/1"), route=2, reverse=True)
        self.assertEqual(selected["start"], choices[1]["end"])
        self.assertEqual(selected["end"], choices[1]["start"])
        ways[1] = way(2, [5, 4], [(0, 0.001), (0.001, 0.001)])
        self.assertEqual(len(routes.chains(ways)), 2)

    def test_sampling_is_unique_ordered_and_reports_missing_stops(self):
        points = [(0, 0), (0, 0.003)]
        route = dict(points=points, way_ids=[1])
        context = [dict(points=points, osm_id=1, name="Test", highway="residential", layer="0")]
        views = [dict(pano_id="a", lat=0, lon=0.0001), dict(pano_id="b", lat=0, lon=0.0029),
                 dict(pano_id="same-position", lat=0, lon=0.0001),
                 dict(pano_id="off-road", lat=0.01, lon=0.001)]
        samples, gaps = routes.match_views(views, route, context, 3)
        self.assertEqual([item["pano_id"] for item in samples], ["a", "b"])
        self.assertEqual([item["stop"] for item in gaps], [2])
        self.assertLess(samples[0]["path_meters"], samples[1]["path_meters"])
        route["points"] = list(reversed(points))
        samples, gaps = routes.match_views(views, route, context, 3)
        self.assertEqual([item["pano_id"] for item in samples], ["b", "a"])
