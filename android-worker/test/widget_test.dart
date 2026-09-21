import "package:flutter_test/flutter_test.dart";

import "package:phonefarm_worker/main.dart";

void main() {
  testWidgets("PhoneFarm Worker renders connection controls", (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const PhoneFarmApp());

    expect(find.text("PhoneFarm Worker"), findsOneWidget);
    expect(find.text("Worker ID"), findsOneWidget);
    expect(find.text("Controller URL"), findsOneWidget);
    expect(find.text("Controller token"), findsOneWidget);
    expect(find.text("Test connection"), findsOneWidget);
    expect(find.text("Start worker"), findsOneWidget);
    expect(find.text("Status: Stopped"), findsOneWidget);
  });
}
